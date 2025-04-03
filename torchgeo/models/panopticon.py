# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Panopticon Foundation Model."""

import timm
import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum


class PanopticonPE(nn.Module):
    """General class defining a patch embedding that takes in arbitrary channel
    dimension and outputs a fixed dimension embedding. This class handles
    the tokenization and projections. The attributed self.chnfus handles
    the channel fusion, i.e. the cross attention over channels.
    """

    def __init__(
        self,
        attn_dim: int,
        embed_dim: int,
        patch_size: int,
        chnfus_cfg: dict = {},
        img_size: int = 224,
    ):
        super().__init__()

        self.conv3d = Conv3dWrapper(patch_size=patch_size, embed_dim=attn_dim)
        self.chnfus = ChnAttn(**chnfus_cfg, dim=attn_dim)
        self.proj = nn.Linear(attn_dim, embed_dim)

        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def forward(self, x_dict: dict) -> Tensor:
        x = x_dict['x']
        chn_ids = x_dict['chn_ids']
        mask = x_dict.get('spec_masks', None)

        x, hp, wp = self.conv3d(x)

        x = self.chnfus(x, chn_ids=chn_ids, mask=mask)  # B,L,D
        x = self.proj(x)

        x = x.reshape(x.shape[0], hp, wp, x.shape[-1])

        return x

    def _init_img_size(self, img_size: int | tuple[int, int]):
        # copied from timm.layers.patch_embed.PatchEmbed._init_img_size
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches


def make_2tuple(x):
    # from dinov2, https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/patch_embed.py
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class Conv3dWrapper(nn.Module):
    """Channel-wise patchification and projection, essentially wrapper around
    1 x P x P conv3d
    """

    def __init__(self, patch_size, embed_dim):
        super().__init__()
        patch_size = make_2tuple(patch_size)
        patch_CHW = (1, *patch_size)
        self.conv3d = nn.Conv3d(1, embed_dim, kernel_size=patch_CHW, stride=patch_CHW)

    def forward(self, x: Tensor):
        x = self.conv3d(x.unsqueeze(1)).squeeze(1)  # B D C Hp Wp
        hp, wp = x.shape[-2:]
        return x.flatten(-2).permute(0, 2, 3, 1), hp, wp  # B C L D


class ChnAttn(nn.Module):
    """Cross attention over channels with channel embeddings to reduce any number
    of channels to a fixed dimension. Inspired by
        https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/arch.py#L185
    """

    def __init__(
        self,
        dim: int,
        chnemb_cfg: dict = {},
        attn_cfg: dict = {},
        layer_norm: bool = False,
    ):
        """Args:
        dim (int): Dimension of the channel attention.
        chnemb_cfg (dict): Key-value pairs for the channel embedding.
        attn_cfg (dict): Key-value pairs for the channel attention.
        layer_norm (bool, optional): Whether to apply layer norm after
            channel attention. Defaults to False.
        """
        super().__init__()

        self.chnemb = ChnEmb(**chnemb_cfg, embed_dim=dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.xattn = CrossAttnNoQueryProj(dim=dim, **attn_cfg)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, chn_ids: Tensor, mask: Tensor = None) -> Tensor:
        """Args:
            x (Tensor): Image tensor of shape (B, C, L, D)
            chn_ids (Tensor): Channel IDs tensor of shape (B,C) or (B,C,2) if
                stds of the SRFs curves are included, see ChnEmb.
            mask (Tensor, optional): Mask tensor of shape (B,C) indicating
                which channels have been masked out. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, L, D) independent of the input
                channel dimension C.
        """
        B, C, L, D = x.shape

        # add embeddings
        chn_embs = self.chnemb(chn_ids)  # B,C,D
        x += chn_embs.unsqueeze(2)

        # abstract away L
        x = x.permute(0, 2, 1, 3).flatten(0, 1)  # BL,C,D
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, L, -1).flatten(0, 1)  # BL,C

        query = self.query.expand(x.shape[0], -1, -1)  # BL,1,D
        assert query.shape == (x.shape[0], 1, x.shape[-1]), (
            f'Expected query to have shape: {x.shape[0], 1, x.shape[-1]}, but got shape: {query.shape}'
        )

        x = self.xattn(query, x, x, key_padding_mask=mask)
        x = x.reshape(B, L, D)

        if hasattr(self, 'layer_norm'):
            return self.layer_norm(x)

        return x


class ChnEmb(torch.nn.Module):
    def __init__(self, embed_dim: int, use_full_spectra=False, opt_coarsity: int = 1):
        """Creates embeddings based on the channel IDs.

        Args:
            embed_dim (int): Embedding dimension.
            use_full_spectra (bool, optional): Whether to additionally to the mean
                also use the standard deviation of optical spectral response (SRF)
                they are provided. This mode only appears in the appendix of the paper.
                Defaults to False.
            opt_coarsity (int, optional): Define the coarsity of how many nanometers
                of the mean SRF are encoded into the same embedding. Defaults to 1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_full_spectra = use_full_spectra
        self.opt_coarsity = opt_coarsity

        dim1 = embed_dim // 3
        dim2 = embed_dim - 2 * dim1
        self.embed_transmit = nn.Parameter(torch.zeros(2, dim1))  # 0:V, 1:H
        self.embed_receive = nn.Parameter(torch.zeros(2, dim1))  # 0:V, 1:H
        self.embed_orbit = nn.Parameter(
            torch.zeros(2, dim2)
        )  # 0:ascending, 1:descending

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim == 2:  # B,C (mus)
            mus = input
        elif input.ndim == 3:  # B,C,2 (mus, sigmas)
            mus = input[:, :, 0]
        sar_indices = mus < 0
        opt_indices = torch.logical_not(sar_indices)
        device = mus.device
        dtype = self.embed_transmit.dtype

        embs = torch.zeros(
            list(mus.shape) + [self.embed_dim], device=device, dtype=dtype
        )

        # build optical embeddings

        mus[opt_indices] = (mus[opt_indices] // self.opt_coarsity).to(mus.dtype)
        if input.ndim == 2 or not self.use_full_spectra:  # only mus
            embs[opt_indices] = get_1d_sincos_pos_embed_from_grid_torch(
                self.embed_dim, mus[opt_indices].view(-1)
            ).to(dtype)

        elif input.ndim == 3:  # full spectra
            raise NotImplementedError('Full spectra not implemented yet.')

        # build sar embeddings

        transmit = torch.cat(
            [self.embed_transmit[0].repeat(2, 1), self.embed_transmit[1].repeat(2, 1)],
            dim=0,
        ).repeat(3, 1)
        receive = torch.cat(
            [
                self.embed_receive[0].unsqueeze(0),
                self.embed_receive[1].repeat(2, 1),
                self.embed_receive[0].unsqueeze(0),
            ],
            dim=0,
        ).repeat(3, 1)
        orbit = torch.stack(
            [self.embed_orbit.mean(axis=0), self.embed_orbit[0], self.embed_orbit[1]]
        ).repeat_interleave(4, dim=0)
        sar_embs = torch.cat([transmit, receive, orbit], dim=1)

        embs[sar_indices] = sar_embs[(-(mus[sar_indices] + 1)).to(torch.int)]

        return embs


class CrossAttnNoQueryProj(nn.Module):
    """Cross Attention without query projection and final projection

    Comment: While doing the final refactor before the release, we noticed that
        we project from patches to 2304 with the conv3d and then again have the
        key & value projections from 2304 to 2304 without non-linearity in-between.
        Hence, the key & value projections are redundant and could be removed,
        significantly reducing the number of parameters. However, this is how
        the paper results were generated and we keep it for reproducibility.
        If you plan on further developing panopticon, please remove the key & value
        projections!
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.scale = head_dim**-0.5

        self.inproj_q = nn.Identity()  # no projection since query is a parameter itself
        self.inproj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.inproj_v = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask=None):
        """q: (B, Nq, D), kv: (B, Nkv, D), key_padding_mask: (B, Nkv)"""
        B, Nq, D = q.shape
        q = (
            self.inproj_q(q)
            .reshape(B, Nq, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        q = q * self.scale

        B, Nkv, D = k.shape  # same as v.shape
        k = (
            self.inproj_k(k)
            .reshape(B, Nkv, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.inproj_v(v)
            .reshape(B, Nkv, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)  # shape: (B, num_heads, Nq, Nkv)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # (B, 1, 1, Nkv)
            attn = attn.masked_fill(key_padding_mask, float('-inf'))

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return x


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


################


class Panopticon_Weights(WeightsEnum):  # type: ignore[misc]
    """Panopticon weights."""

    VIT_BASE14 = Weights(
        url='https://huggingface.co/lewaldm/panopticon/resolve/main/panopticon_vitb14_teacher.pth',
        transforms=None,
        meta={
            'model': 'panopticon_vitb14',
            'publication': 'https://arxiv.org/abs/2503.10845',
            'repo': 'https://github.com/Panopticon-FM/panopticon',
            'ssl_method': 'dinov2+spectral_progressive_pretraining',
        },
    )


class Panopticon(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dinov2_vit = timm.create_model('vit_base_patch14_dinov2', dynamic_img_size=True)

        dinov2_vit.patch_embed = PanopticonPE(
            attn_dim=2304,
            embed_dim=768,
            patch_size=14,
            img_size=518,
            chnfus_cfg={'attn_cfg': {'num_heads': 16}},
        )

        self.model = dinov2_vit

    def forward(self, x: Tensor, chn_ids: Tensor) -> Tensor:
        """Forward pass of the model including forward pass through the head.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
            chn_ids (Tensor): Tensor of shape (B,C) encoding the spectral information
                of each channel. For optical channels, this is the wavelength
                in nanometers. For SAR channels, this is a negative integer as
                outlined in https://github.com/Panopticon-FM/panopticon/blob/main/dinov2/configs/data/satellites/sentinel1.yaml
        Returns:
            Tensor: Output.
        """
        return self.model(dict(x=x, chn_ids=chn_ids))


def panopticon_vitb14(weights: Panopticon_Weights | None = None) -> torch.nn.Module:
    """Panopticon ViT-Base model.

    Panopticon can handle arbitrary optical channel and SAR combinations.
    While image sizes up to 518x518 are possible, please match the training setting of 224x224 images
    with a patch size of 14. For more information on how to use the model,
    see https://github.com/Panopticon-FM/panopticon?tab=readme-ov-file#using-panopticon.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.10845

    .. versionadded:: 0.7

    Returns:
        The Panopticon ViT-Base model with the published weights loaded.
    """
    model = Panopticon()

    if weights:
        state_dict = weights.get_state_dict(progress=True)
        state_dict.pop('mask_token')
        model.model.load_state_dict(state_dict, strict=True)

    return model
