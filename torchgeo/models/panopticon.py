# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Panopticon Foundation Model."""

from typing import Any, overload

import timm
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

from ..samplers.utils import _to_tuple
from .copernicusfm import resize_abs_pos_embed


class PanopticonPE(nn.Module):
    """Defines the Panopticon Patch Embedding."""

    def __init__(
        self,
        attn_dim: int,
        embed_dim: int,
        patch_size: int,
        chnfus_cfg: dict[str, Any] = {},
        img_size: int = 224,
    ) -> None:
        """Initialize a new Panopticon instance.

        Args:
            attn_dim: Embedding dimension on which the channel attention operates.
            embed_dim: Dimension of embeddings that the PanopticonPE outputs.
            patch_size: The patch size.
            chnfus_cfg: Key word arguemnts defining the channel attention.
            img_size: Image size.
        """
        super().__init__()

        self.conv3d = Conv3dWrapper(patch_size=patch_size, embed_dim=attn_dim)
        self.chnfus = ChnAttn(**chnfus_cfg, dim=attn_dim)
        self.proj = nn.Linear(attn_dim, embed_dim)

        self.patch_size: tuple[int, int] = _to_tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def forward(self, x_dict: dict[str, Tensor]) -> Tensor:
        """Forward pass of the model.

        Args:
            x_dict: Dictionary with inputs to the model. Contains keys 'imgs' with
                tensor of shape (B, C, H, W) and 'chn_ids' with tensor of shape (B,C)
                encoding the channel ids.

        Returns:
            Output of shape (B, num_patches, embed_dim).
        """
        x: Tensor = x_dict['imgs']
        chn_ids = x_dict['chn_ids']
        mask = x_dict.get('spec_masks', None)

        x, hp, wp = self.conv3d(x)

        x = self.chnfus(x, chn_ids=chn_ids, mask=mask)  # B,L,D
        x = self.proj(x)

        return x

    @overload
    def _init_img_size(self, img_size: None) -> tuple[None, None, None]: ...

    @overload
    def _init_img_size(
        self, img_size: int | tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int], int]: ...

    def _init_img_size(
        self, img_size: None | int | tuple[int, int]
    ) -> tuple[None, None, None] | tuple[tuple[int, int], tuple[int, int], int]:
        """Compute the image size, grid size and number of patches.

        Args:
            img_size: Image size.

        Returns:
            Image size tuple, grid size tuple, and number of patches.
        """
        # copied from timm.layers.patch_embed.PatchEmbed._init_img_size (1.0.10)
        assert self.patch_size
        if img_size is None:
            return None, None, None
        tuple_img_size = _to_tuple(img_size)
        grid_size = (
            tuple_img_size[0] // self.patch_size[0],
            tuple_img_size[1] // self.patch_size[1],
        )
        num_patches = grid_size[0] * grid_size[1]
        return tuple_img_size, grid_size, num_patches


class Conv3dWrapper(nn.Module):
    """Channel-wise patchification and projection."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        """Initialize a conv3d wrapper.

        Args:
            patch_size (int): Patch size.
            embed_dim (int): Embedding dimension.
        """
        super().__init__()
        tuple_patch_size = _to_tuple(patch_size)
        patch_CHW = (1, tuple_patch_size[0], tuple_patch_size[1])
        self.conv3d = nn.Conv3d(1, embed_dim, kernel_size=patch_CHW, stride=patch_CHW)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Forward pass.

        Args:
            x (Tensor): Tensor of shape (B, C, H, W) where B is the batch size,

        Returns:
            Tensor:
            hp: number of patches in heigth
            wp: number of patches in width
        """
        x = self.conv3d(x.unsqueeze(1)).squeeze(1)  # B D C Hp Wp
        hp, wp = x.shape[-2:]
        return x.flatten(-2).permute(0, 2, 3, 1), hp, wp  # B C L D


class ChnAttn(nn.Module):
    """Cross attention over channels with channel embeddings.

    Can reduce any number of channels to a fixed dimension. Inspired by
        https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/arch.py#L185
    """

    def __init__(
        self,
        dim: int,
        chnemb_cfg: dict[str, Any] = {},
        attn_cfg: dict[str, Any] = {},
        layer_norm: bool = False,
    ) -> None:
        """Initialize a channel attention module.

        Args:
            dim (int): Dimension of the channel attention.
            chnemb_cfg (dict, optional): Key-value pairs for the channel embedding. Defaults to {}.
            attn_cfg (dict, optional): Key-value pairs for the channel attention. Defaults to {}.
            layer_norm (bool, optional): Whether to apply layer norm after
                channel attention. Defaults to False.
        """
        super().__init__()

        self.chnemb = ChnEmb(**chnemb_cfg, embed_dim=dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.xattn = CrossAttnNoQueryProj(dim=dim, **attn_cfg)

        if layer_norm:
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, chn_ids: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Image tensor of shape (B, C, L, D)
            chn_ids (Tensor): Channel IDs tensor of shape (B,C), see ChnEmb.
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
            x = self.layer_norm(x)

        return x


class ChnEmb(torch.nn.Module):
    """Computes embeddings from Channel IDs."""

    def __init__(
        self, embed_dim: int, use_full_spectra: bool = False, opt_coarsity: int = 1
    ) -> None:
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
        """Forward pass.

        Args:
            input (Tensor): Tensor of shape (B,C) or (B,C,2). If (B,C), we expect
                the channel IDs. If (B,C,2), we expect the channel IDs and the
                standard deviation of the spectral response function (SRF).

        Returns:
            Tensor: Embeddings of shape (B,C,embed_dim).
        """
        if input.ndim == 2:  # B,C (mus)
            mus = input
        elif input.ndim == 3:  # B,C,2 (mus, sigmas)
            mus = input[:, :, 0]
        sar_indices = mus < 0
        opt_indices = torch.logical_not(sar_indices)
        device = mus.device
        dtype = self.embed_transmit.dtype

        embs = torch.zeros(
            [*list(mus.shape), self.embed_dim], device=device, dtype=dtype
        )

        # build optical embeddings

        mus[opt_indices] = (mus[opt_indices] // self.opt_coarsity).to(mus.dtype)
        if input.ndim == 2 or not self.use_full_spectra:  # only mus
            embs[opt_indices] = get_1d_sincos_pos_embed_from_grid_torch(
                self.embed_dim, mus[opt_indices].view(-1)
            ).to(dtype)

        elif input.ndim == 3:  # full spectra
            mus_opt = mus[opt_indices]
            sigmas_opt = input[opt_indices][:, 1]
            embs[opt_indices] = get_1d_sincos_ipe_analytical(
                mus_opt, sigmas_opt, self.embed_dim, device
            ).to(dtype)

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
            [self.embed_orbit.mean(dim=0), self.embed_orbit[0], self.embed_orbit[1]]
        ).repeat_interleave(4, dim=0)
        sar_embs = torch.cat([transmit, receive, orbit], dim=1)

        embs[sar_indices] = sar_embs[(-(mus[sar_indices] + 1)).to(torch.int)]

        return embs


class CrossAttnNoQueryProj(nn.Module):
    """Cross Attention without query projection and final projection."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        """Initialize a cross attention module.

        Args:
            dim (int): dimension of attention.
            num_heads (int, optional): number of heads. Defaults to 8.
            qkv_bias (bool, optional): whether to use query, key, and
                value biases. Defaults to False.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.scale = head_dim**-0.5

        self.inproj_q = nn.Identity()  # no projection since query is a parameter itself
        self.inproj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.inproj_v = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            q (Tensor): query of shape (B, Nq, D)
            k (Tensor): key tensor of shape (B, Nkv, D)
            v (Tensor): value tensor of shape (B, Nkv, D)
            key_padding_mask (Tensor, optional): key padding mask tensor of shape (B, Nkv). Defaults to None.

        Returns:
            Tensor: resulting tensor
        """
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


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim: int, pos: Tensor) -> Tensor:
    """Generate standard sin cos positional embeddings.

    Args:
        embed_dim (int): output dimension for each position
        pos (Tensor): a list of positions to be encoded: size (M,)

    Returns:
        Tensor: Tensor of embeddings of shape (M,D)
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


def get_1d_sincos_ipe_analytical(
    mu: Tensor, sigma: Tensor, D: int, device: torch.device, temperature: int = 10000
) -> Tensor:
    """Compute the integrated positional embedding (IPE).

    This is only used in the appendix of the paper. You can find further
    information on the motivation & formulas there.

    Args:
        mu (Tensor): Tensor containing the mus.
        sigma (Tensor): Tensor containing the sigmas.
        D (int): dimension of the embeddings.
        device (torch.device): Torch device to move the embeddings to.
        temperature (int, optional): temperature of embeddings. Defaults to 10000.

    Returns:
        Tensor: Tensor of embeddings.
    """
    # Create meshgrid for vectorized computation
    d_mesh = torch.arange(D, dtype=torch.float32, device=device)
    mu_mesh = mu.unsqueeze(1).expand(-1, D)
    sigma_mesh = sigma.unsqueeze(1).expand(-1, D)

    # Compute frequencies omega_i
    omega = 1.0 / (temperature ** (2 * d_mesh / D))

    # Compute the Gaussian decay term for each frequency
    # Note: We divide by sigma to normalize similar to how a Gaussian kernel would be normalized
    gaussian_term = torch.exp(-0.5 * (omega.unsqueeze(0) * sigma_mesh) ** 2)

    # Compute sine and cosine terms
    sin_term = torch.sin(omega.unsqueeze(0) * mu_mesh)
    cos_term = torch.cos(omega.unsqueeze(0) * mu_mesh)

    # Combine based on even/odd indices
    IPE = torch.where(
        d_mesh % 2 == 0,
        gaussian_term * sin_term,  # even indices
        gaussian_term * cos_term,
    )  # odd indices

    return IPE


class Panopticon(torch.nn.Module):
    """Panopticon ViT-Base Foundation Model."""

    def __init__(
        self, attn_dim: int = 2304, embed_dim: int = 768, img_size: int = 224
    ) -> None:
        """Initialize a panopticon model.

        Args:
            attn_dim (int, optional): Dimension of channel attention. Defaults to 2304.
            embed_dim (int, optional): Embedding dimension of backbone. Defaults to 768.
            img_size (int, optional): Image size. Panopticon can be initizialized
                with any image size but image size is fixed after initialization.
                For optimal performance, we recommend to use the same image size
                as used during training. For the published weights, this is 224.
                Defaults to 224.
        """
        super().__init__()
        dinov2_vit = timm.create_model('vit_base_patch14_dinov2')
        patch_size = 14

        dinov2_vit.patch_embed = PanopticonPE(
            attn_dim=attn_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            chnfus_cfg={'attn_cfg': {'num_heads': 16}},
        )
        dinov2_vit.pos_embed = torch.nn.Parameter(
            torch.randn(1, 1 + (img_size // patch_size) ** 2, embed_dim)
        )

        # self.model: timm.models.vision_transformer.VisionTransformer = dinov2_vit
        self.model: nn.Module = dinov2_vit

    def forward(self, x_dict: dict[str, Any]) -> Tensor:
        """Forward pass of the model including forward pass through the head.

        Args:
            x_dict (dict): Dictionary with keys:
                imgs (Tensor): Input tensor of shape (B, C, H, W).
                chn_ids (Tensor): Tensor of shape (B,C) encoding the spectral information
                    of each channel. For optical channels, this is the wavelength
                    in nanometers. For SAR channels, this is a negative integer as
                    outlined in https://github.com/Panopticon-FM/panopticon/blob/main/dinov2/configs/data/satellites/sentinel1.yaml
        Returns:
            Tensor: Embeddings.
        """
        out: Tensor = self.model.forward(x_dict)
        return out


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


def panopticon_vitb14(
    weights: Panopticon_Weights | None = None, img_size: int = 224, **kwargs: int
) -> torch.nn.Module:
    """Panopticon ViT-Base model.

    Panopticon can handle arbitrary optical channel and SAR combinations.
    It can also be initialized with any image size where the image size is
    fixed after initialization. However, we recommend to set 224 in alignment
    with the pretraining. For more information on how to use the model,
    see https://github.com/Panopticon-FM/panopticon?tab=readme-ov-file#using-panopticon.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.10845

    .. versionadded:: 0.7

    Returns:
        The Panopticon ViT-Base model with the published weights loaded.
    """
    model = Panopticon(img_size=img_size, **kwargs)
    patch_size = 14  # fixed

    if weights:
        state_dict = weights.get_state_dict(progress=True)
        state_dict.pop('mask_token')

        # interpolate positional embeddings (timm==0.9.2) does not support this yet
        state_dict['pos_embed'] = resize_abs_pos_embed(
            state_dict['pos_embed'], img_size // patch_size, 518 // patch_size
        )

        model.model.load_state_dict(state_dict, strict=True)

    return model
