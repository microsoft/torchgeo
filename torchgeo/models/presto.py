# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
# Copyright (c) 2024 Presto Authors

# Modified from https://github.com/nasaharvest/presto

"""Pretrained Remote Sensing Transformer (Presto)."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from timm.models.vision_transformer import Block
from torchvision.models._api import Weights, WeightsEnum

BANDS_GROUPS_IDX: dict[str, Sequence[int]] = {
    'S1': (0, 1),
    'S2_RGB': (2, 3, 4),
    'S2_Red_Edge': (5, 6, 7),
    'S2_NIR_10m': (8,),
    'S2_NIR_20m': (9,),
    'S2_SWIR': (10, 11),
    'ERA5': (12, 13),
    'SRTM': (14, 15),
    'NDVI': (16,),
}
NUM_DYNAMIC_WORLD_CLASSES = 9


def get_sinusoid_encoding_table(
    positions: int | list[int], device: torch.device, d_hid: int, T: int = 1000
) -> torch.Tensor:
    """Generate a sinusoid positional encoding table for the given positions.

    Args:
        positions: Either an integer specifying the maximum position (encoded as
            ``range(positions)``) or a list of integer positions to encode.
        device: The device on which to place the returned tensor.
        d_hid: The dimensionality of the positional encoding.
        T: Scaling factor that controls the frequencies of the sinusoidal basis
            functions.

    Returns:
        A tensor of shape ``(len(positions), d_hid)`` containing the positional
        encodings.
    """
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position: int, hid_idx: int) -> float:
        """Compute the angle for a single position/index pair."""
        return float(position / np.power(T, 2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position: int) -> list[float]:
        """Build the angle vector for a single position."""
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    sinusoid_table_tensor = torch.from_numpy(sinusoid_table.astype(float))
    sinusoid_table_tensor = sinusoid_table_tensor.to(torch.float).to(device)
    return sinusoid_table_tensor


def get_month_encoding_table(d_hid: int, device: torch.device) -> torch.Tensor:
    """Sinusoid month encoding table, for 12 months indexed from 0-11.

    Args:
        d_hid: Dimension of the hidden state.
        device: Device to place the tensor on.

    Returns:
        A tensor of shape (12, d_hid) containing the month encoding.
    """
    assert d_hid % 2 == 0
    angles = np.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = np.sin(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    cos_table = np.cos(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    month_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1).astype(
        float
    )
    month_table_tensor: torch.Tensor = (
        torch.from_numpy(month_table).to(torch.float).to(device)
    )
    return month_table_tensor


def month_to_tensor(
    month: torch.Tensor | int, batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Convert month indices into a per-sample sequence, wrapping every 12 months.

    Args:
        month: Month as an integer, per-sample start month tensor of shape [batch],
            or explicit month tensor of shape [batch, seq_len].
        batch_size: Number of samples in the batch.
        seq_len: Length of the sequence.
        device: Device to place the tensor on.

    Returns:
        A tensor of shape (batch_size, seq_len) containing the month encoding.
    """
    if isinstance(month, int):
        assert month < 12
    else:
        assert month.max().item() < 12

    if isinstance(month, int):
        # >>> torch.fmod(torch.tensor([9., 10, 11, 12, 13, 14]), 12)
        # tensor([ 9., 10., 11.,  0.,  1.,  2.])
        month = (
            torch.fmod(torch.arange(month, month + seq_len, dtype=torch.long), 12)
            .expand(batch_size, seq_len)
            .to(device)
        )
    elif len(month.shape) == 1:
        month = torch.stack(
            [
                torch.fmod(torch.arange(start=m, end=m + seq_len, dtype=torch.long), 12)
                for m in month
            ]
        ).to(device)
    return month


class Encoder(nn.Module):
    """Encoder for the Presto model."""

    def __init__(
        self,
        band_groups: dict[str, Sequence[int]] | None = None,
        embedding_size: int = 128,
        channel_embed_ratio: float = 0.25,
        month_embed_ratio: float = 0.25,
        depth: int = 2,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        max_sequence_length: int = 24,
    ) -> None:
        """Initialize a new Encoder instance.

        Args:
            band_groups: Mapping of band group names to channel indices.
            embedding_size: Size of the embedding for each token.
            channel_embed_ratio: Ratio of the embedding size to use for channel embeddings.
            month_embed_ratio: Ratio of the embedding size to use for month embeddings.
            depth: Number of Transformer blocks in the encoder.
            mlp_ratio: Ratio of the hidden dimension in the MLP compared to the embedding size.
            num_heads: Number of attention heads in each Transformer block.
            max_sequence_length: Maximum length of the input sequence.
        """
        super().__init__()

        self.band_groups = (
            dict(band_groups) if band_groups is not None else BANDS_GROUPS_IDX
        )
        self.embedding_size = embedding_size

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx
            for idx, (group_name, _) in enumerate(self.band_groups.items())
        }
        self.band_group_to_idx['dynamic_world'] = (
            max(self.band_group_to_idx.values()) + 1
        )

        self.eo_patch_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.band_groups.items()
            }
        )
        self.dw_embed = nn.Embedding(
            num_embeddings=NUM_DYNAMIC_WORLD_CLASSES + 1, embedding_dim=embedding_size
        )
        self.latlon_embed = nn.Linear(3, embedding_size)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)

        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        pos_embedding_size = int(
            embedding_size * (1 - (channel_embed_ratio + month_embed_ratio))
        )
        channel_embedding_size = int(embedding_size * channel_embed_ratio)
        month_embedding_size = int(embedding_size * month_embed_ratio)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, pos_embedding_size), requires_grad=False
        )
        month_tab = get_month_encoding_table(
            d_hid=month_embedding_size, device=self.pos_embed.device
        )
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        self.channel_embed = nn.Embedding(
            num_embeddings=len(self.band_groups) + 1,
            embedding_dim=channel_embedding_size,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize the weights of the encoder."""
        pos_embed = get_sinusoid_encoding_table(
            positions=self.pos_embed.shape[1],
            device=self.pos_embed.device,
            d_hid=self.pos_embed.shape[-1],
            T=1000,
        )
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for nn.Linear and nn.LayerNorm.

        Args:
            m: The module to initialize.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def cartesian(latlons: torch.Tensor) -> torch.Tensor:
        """Convert latitude and longitude to Cartesian coordinates.

        Args:
            latlons: Tensor of shape [batch, 2] containing latitude and longitude in degrees.

        Returns:
            Tensor of shape [batch, 3] containing Cartesian coordinates (x, y, z).
        """
        with torch.no_grad():
            # an embedding is calculated for all timesteps. This is then expanded
            # for each timestep in the sequence
            latlon_radians = latlons * math.pi / 180
            lats, lons = latlon_radians[:, 0], latlon_radians[:, 1]
            x = torch.cos(lats) * torch.cos(lons)
            y = torch.cos(lats) * torch.sin(lons)
            z = torch.sin(lats)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def mask_tokens(
        x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask tokens in the input tensor.

        Args:
            x: Input tensor of shape [batch, timesteps, channels].
            mask: Mask tensor of shape [batch, timesteps, channels].

        Returns:
            The masked tensor, kept indices, and removed indices.
        """
        summed = mask.sum(
            dim=(1, 2)
        )  # summed tells me the number of masked elements per batch idx
        assert summed.max() == summed.min(), f'{summed.max()}, {summed.min()}'

        batch_size = x.shape[0]
        removed_elements_per_batch = int(summed.max() / mask.shape[2])
        kept_elements_per_batch = x.shape[1] - removed_elements_per_batch
        embedding_dim = x.shape[-1]

        # we want the mask to just be the indices of the masked tokens
        indices = repeat(
            torch.arange(0, x.shape[1]).long().to(x.device), 'd -> b d', b=x.shape[0]
        )

        x = x[~mask.bool()].view(batch_size, kept_elements_per_batch, embedding_dim)

        mask = mask[:, :, 0]
        kept_indices = indices[~mask.bool()].view(batch_size, kept_elements_per_batch)
        removed_indices = indices[mask.bool()].view(
            batch_size, removed_elements_per_batch
        )

        return x, kept_indices, removed_indices

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: torch.Tensor | None = None,
        month: torch.Tensor | int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.

        Args:
            x: Input tensor of shape [batch, timesteps, channels].
            dynamic_world: Dynamic world tensor of shape [batch, timesteps].
            latlons: Latitude and longitude tensor of shape [batch, 2].
            mask: Mask tensor of shape [batch, timesteps, channels]. Defaults to None.
            month: Month tensor or integer representing the month. Defaults to 0.

        Returns:
            Tuple containing the encoded tensor, kept indices, and removed indices.
        """
        device = x.device

        if mask is None:
            mask = torch.zeros_like(x, device=x.device).to(x.dtype)
        else:
            mask = mask.to(device=x.device).to(x.dtype)

        months = month_to_tensor(month, x.shape[0], x.shape[1], device)
        month_embedding = self.month_embed(months)
        positional_embedding = repeat(
            self.pos_embed[:, : x.shape[1], :],
            'b t d -> (repeat b) t d',
            repeat=x.shape[0],
        )

        # we assume the number of masked patches is the same
        # for all items in the batch. Otherwise things become a headache
        all_tokens, all_masks = [], []

        for channel_group, channel_idxs in self.band_groups.items():
            tokens = self.eo_patch_embed[channel_group](x[:, :, channel_idxs])
            channel_embedding = self.channel_embed(
                torch.tensor(self.band_group_to_idx[channel_group]).long().to(device)
            )
            channel_embedding = repeat(
                channel_embedding, 'd -> b t d', b=x.shape[0], t=x.shape[1]
            )
            if channel_group == 'SRTM':
                # for SRTM, we reduce it to a single token instead of
                # a token per timestep
                channel_wise_positional_embedding = torch.cat(
                    (
                        torch.zeros_like(month_embedding[:, 0:1]),
                        channel_embedding[:, 0:1],
                        torch.zeros_like(positional_embedding[:, 0:1]),
                    ),
                    dim=-1,
                )
                indices = slice(0, 1)
            else:
                channel_wise_positional_embedding = torch.cat(
                    (month_embedding, channel_embedding, positional_embedding), dim=-1
                )
                indices = slice(None)

            tokens = tokens[:, indices]
            tokens += channel_wise_positional_embedding
            all_tokens.append(tokens)
            group_mask = repeat(
                torch.max(mask[:, indices, channel_idxs], dim=-1)[0],
                'b t -> b t d',
                d=tokens.shape[-1],
            )
            all_masks.append(group_mask)

        # then, dynamic world
        tokens = self.dw_embed(dynamic_world)
        channel_embedding = self.channel_embed(
            torch.tensor(self.band_group_to_idx['dynamic_world']).long().to(device)
        )
        channel_embedding = repeat(
            channel_embedding, 'd -> b t d', b=x.shape[0], t=x.shape[1]
        )
        positional_embedding = torch.cat(
            (month_embedding, channel_embedding, positional_embedding), dim=-1
        )
        tokens += positional_embedding
        all_tokens.append(tokens)

        # now we calculate the mask for these [b, t] tokens
        group_mask = repeat(
            dynamic_world == NUM_DYNAMIC_WORLD_CLASSES,
            'b t -> b t d',
            d=tokens.shape[-1],
        )
        all_masks.append(group_mask)

        x = torch.cat(all_tokens, dim=1)  # [batch, timesteps, embedding_dim]
        mask = torch.cat(all_masks, dim=1)  # [batch, timesteps, embedding_dim]
        x, kept_indices, removed_indices = self.mask_tokens(x, mask)

        # append latlon tokens
        latlon_tokens = self.latlon_embed(self.cartesian(latlons)).unsqueeze(1)
        x = torch.cat((latlon_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # mask will be a boolean of shape [batch, total_num_tokens]
        return self.norm(x), kept_indices, removed_indices


class Decoder(nn.Module):
    """Decoder for the Presto model."""

    def __init__(
        self,
        channel_embeddings: nn.Embedding,
        band_groups: dict[str, Sequence[int]] | None = None,
        encoder_embed_dim: int = 128,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        mlp_ratio: int = 2,
        max_sequence_length: int = 24,
    ) -> None:
        """Initialize a new Decoder instance.

        Args:
            channel_embeddings: Embedding layer for channel groups.
            band_groups: Mapping of band group names to channel indices.
            encoder_embed_dim: Embedding dimension of the encoder.
            decoder_embed_dim: Embedding dimension of the decoder.
            decoder_depth: Number of Transformer blocks in the decoder.
            decoder_num_heads: Number of attention heads in each Transformer block.
            mlp_ratio: Ratio of the hidden dimension in the MLP compared to the embedding size.
            max_sequence_length: Maximum length of the input sequence.
        """
        super().__init__()

        self.band_groups = (
            dict(band_groups) if band_groups is not None else BANDS_GROUPS_IDX
        )

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx
            for idx, (group_name, _) in enumerate(self.band_groups.items())
        }
        self.band_group_to_idx['dynamic_world'] = (
            max(self.band_group_to_idx.values()) + 1
        )

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.eo_decoder_pred = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embed_dim, len(group))
                for group_name, group in self.band_groups.items()
            }
        )
        self.dw_decoder_pred = nn.Linear(decoder_embed_dim, NUM_DYNAMIC_WORLD_CLASSES)

        self.channel_embeddings = channel_embeddings
        channel_embedding_dims = channel_embeddings.weight.shape[-1]
        remaining_embeddings = decoder_embed_dim - channel_embedding_dims
        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, int(remaining_embeddings) // 2),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(
            d_hid=int(remaining_embeddings) // 2, device=self.pos_embed.device
        )
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize the weights of the decoder."""
        pos_embed = get_sinusoid_encoding_table(
            positions=self.pos_embed.shape[1],
            device=self.pos_embed.device,
            d_hid=self.pos_embed.shape[-1],
            T=1000,
        )
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights for nn.Linear and nn.LayerNorm.

        Args:
            m: The module to initialize.
        """
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_masked_tokens(
        self, x: torch.Tensor, kept_indices: torch.Tensor, removed_indices: torch.Tensor
    ) -> torch.Tensor:
        """Add masked tokens to the input tensor.

        Args:
            x: Input tensor of shape [batch, timesteps, embedding_dim].
            kept_indices: Indices of the kept tokens.
            removed_indices: Indices of the removed tokens.

        Returns:
            Tensor with masked tokens added.
        """
        mask_tokens = repeat(
            self.mask_token, 'd -> b t d', b=x.shape[0], t=removed_indices.shape[1]
        )

        x = torch.cat([x, mask_tokens], dim=1)

        # sort according to their indices. Shape is [batch, index]
        combined_indices = torch.cat([kept_indices, removed_indices], dim=1) + 1
        # 0 for latlon index
        combined_indices = torch.sort(
            torch.cat(
                [torch.zeros_like(combined_indices[:, 0:1]), combined_indices], dim=1
            )
        )[1]
        # and then tile for each dimension
        combined_indices = repeat(combined_indices, 'b t -> b t d', d=x.shape[-1])
        x = torch.gather(x, 1, combined_indices)
        return x

    def add_embeddings(
        self, x: torch.Tensor, month: torch.Tensor | int
    ) -> torch.Tensor:
        """Add positional and month embeddings to the input tensor.

        Args:
            x: Input tensor of shape [batch, timesteps, embedding_dim].
            month: Month tensor or integer representing the month.

        Returns:
            Tensor with positional and month embeddings added.
        """
        num_channel_groups = len(self.band_group_to_idx)
        # -2 since we remove srtm and latlon, and -1 since the srtm
        # channel group doesn't have timesteps
        num_timesteps = int((x.shape[1] - 2) / (num_channel_groups - 1))
        srtm_index = self.band_group_to_idx['SRTM'] * num_timesteps
        months = month_to_tensor(month, x.shape[0], num_timesteps, x.device)

        # when we expand the encodings, each channel_group gets num_timesteps
        # encodings. However, there is only one SRTM token so we remove the
        # excess SRTM encodings
        device = x.device
        remove_mask = torch.full(
            size=(num_timesteps * num_channel_groups,),
            fill_value=False,
            device=device,
            dtype=torch.bool,
        )
        remove_indices = torch.arange(num_timesteps - 1, device=device) + srtm_index
        remove_mask[remove_indices] = True

        month_embedding = repeat(
            self.month_embed(months),
            'b t d -> b (repeat t) d',
            repeat=num_channel_groups,
        )
        month_embedding = month_embedding[:, ~remove_mask]
        month_embedding[:, srtm_index] = 0

        positional_embedding = repeat(
            self.pos_embed[:, :num_timesteps, :],
            'b t d -> (b2 b) (t2 t) d',
            b2=x.shape[0],
            t2=num_channel_groups,
        )
        positional_embedding = positional_embedding[:, ~remove_mask]
        positional_embedding[:, srtm_index] = 0

        channel_embeddings = torch.repeat_interleave(
            self.channel_embeddings.weight, repeats=num_timesteps, dim=0
        )
        channel_embeddings = repeat(channel_embeddings, 'c d -> b c d', b=x.shape[0])
        channel_embeddings = channel_embeddings[:, ~remove_mask]

        positional_embedding = torch.cat(
            (month_embedding, channel_embeddings, positional_embedding), dim=-1
        )

        # add the zero embedding for the latlon token
        positional_embedding = torch.cat(
            [torch.zeros_like(positional_embedding[:, 0:1, :]), positional_embedding],
            dim=1,
        )

        x += positional_embedding
        return x

    def reconstruct_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reconstruct the inputs from the decoder output.

        Args:
            x: Output tensor of shape [batch, timesteps, embedding_dim].

        Returns:
            Tuple containing the reconstructed inputs for each channel group and the dynamic world output.
        """
        # remove the latlon token
        x = x[:, 1:, :]

        # split into channel groups
        num_channel_groups = len(self.band_group_to_idx) - 1
        num_timesteps = int((x.shape[1] - 1) / num_channel_groups)
        srtm_index = self.band_group_to_idx['SRTM'] * num_timesteps
        srtm_token = x[:, srtm_index : srtm_index + 1, :]

        mask = torch.full((x.shape[1],), True, device=x.device)
        mask[srtm_index] = False
        x = x[:, mask]

        x = x.view(x.shape[0], num_channel_groups, num_timesteps, x.shape[-1])

        eo_output, dw_output = [], None
        for group_name, idx in self.band_group_to_idx.items():
            if group_name == 'SRTM':
                eo_output.append(
                    repeat(
                        self.eo_decoder_pred[group_name](srtm_token),
                        'b t d -> b (t2 t) d',
                        t2=num_timesteps,
                    )
                )
            else:
                if idx > self.band_group_to_idx['SRTM']:
                    idx -= 1
                group_tokens = x[:, idx]
                if group_name == 'dynamic_world':
                    dw_output = self.dw_decoder_pred(group_tokens)
                else:
                    eo_output.append(self.eo_decoder_pred[group_name](group_tokens))

        # we can just do this concatenation because the BANDS_GROUP_IDX is ordered
        return torch.cat(eo_output, dim=-1), dw_output

    def forward(
        self,
        x: torch.Tensor,
        kept_indices: torch.Tensor,
        removed_indices: torch.Tensor,
        month: torch.Tensor | int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the decoder.

        Args:
            x: Input tensor of shape [batch, timesteps, embedding_dim].
            kept_indices: Indices of the kept tokens.
            removed_indices: Indices of the removed tokens.
            month: Month tensor or integer representing the month. Defaults to 0.

        Returns:
            Tuple containing the reconstructed inputs for each channel group and the dynamic world output.
        """
        x = self.decoder_embed(x)
        x = self.add_masked_tokens(x, kept_indices, removed_indices)
        x = self.add_embeddings(x, month)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        reconstructed_inputs, dw_output = self.reconstruct_inputs(x)
        return reconstructed_inputs, dw_output


class Presto(nn.Module):
    """Pretrained Remote Sensing Transformer (Presto).

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        band_groups: dict[str, Sequence[int]] | None = None,
        encoder_embedding_size: int = 128,
        channel_embed_ratio: float = 0.25,
        month_embed_ratio: float = 0.25,
        encoder_depth: int = 2,
        mlp_ratio: int = 4,
        encoder_num_heads: int = 8,
        decoder_embedding_size: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 8,
        max_sequence_length: int = 24,
    ) -> None:
        """Initialize a new Presto instance.

        Args:
            band_groups: Mapping of band group names to channel indices.
            encoder_embedding_size: Size of the embedding for each token in the encoder.
            channel_embed_ratio: Ratio of the embedding size to use for channel embeddings in the encoder.
            month_embed_ratio: Ratio of the embedding size to use for month embeddings in the encoder.
            encoder_depth: Number of Transformer blocks in the encoder.
            mlp_ratio: Ratio of the hidden dimension in the MLP compared to the embedding size in the encoder.
            encoder_num_heads: Number of attention heads in each Transformer block in the encoder.
            decoder_embedding_size: Size of the embedding for each token in the decoder.
            decoder_depth: Number of Transformer blocks in the decoder.
            decoder_num_heads: Number of attention heads in each Transformer block in the decoder.
            max_sequence_length: Maximum length of the input sequence.
        """
        super().__init__()
        self.encoder = Encoder(
            band_groups=band_groups,
            embedding_size=encoder_embedding_size,
            channel_embed_ratio=channel_embed_ratio,
            month_embed_ratio=month_embed_ratio,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            num_heads=encoder_num_heads,
            max_sequence_length=max_sequence_length,
        )
        decoder_band_groups = self.encoder.band_groups
        self.decoder = Decoder(
            channel_embeddings=self.encoder.channel_embed,
            band_groups=decoder_band_groups,
            encoder_embed_dim=encoder_embedding_size,
            decoder_embed_dim=decoder_embedding_size,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: torch.Tensor | None = None,
        month: torch.Tensor | int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the Presto model.

        Args:
            x: Input tensor of shape [batch, timesteps, channels].
            dynamic_world: Dynamic world tensor of shape [batch, timesteps].
            latlons: Latitude and longitude tensor of shape [batch, 2].
            mask: Mask tensor of shape [batch, timesteps, channels]. Defaults to None.
            month: Month tensor or integer representing the month. Defaults to 0.

        Returns:
            Tuple containing the reconstructed inputs for each channel group and the dynamic world output.
        """
        x, kept_indices, removed_indices = self.encoder(
            x=x, dynamic_world=dynamic_world, latlons=latlons, mask=mask, month=month
        )
        reconstructed_inputs, dw_output = self.decoder(
            x, kept_indices, removed_indices, month
        )
        return reconstructed_inputs, dw_output


class Presto_Weights(WeightsEnum):
    """Presto weights.

    .. versionadded:: 0.9
    """

    PRESTO = Weights(
        url='https://hf.co/torchgeo/presto/resolve/40de9c69b1611bb11de7b572cf3d24bb60cb8c82/model-bfa691d3.pth',
        transforms=nn.Identity(),
        meta={
            'dataset': 'LEM (Presto pretraining dataset)',
            'model': 'Presto',
            'publication': 'https://arxiv.org/abs/2304.14065',
            'repo': 'https://github.com/nasaharvest/presto',
        },
    )


def presto(weights: Presto_Weights | None = None, *args: Any, **kwargs: Any) -> Presto:
    """Presto model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2304.14065

    .. versionadded:: 0.9

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`Presto`.
        **kwargs: Additional keyword arguments to pass to :class:`Presto`.

    Returns:
        A Presto model.
    """
    model = Presto(*args, **kwargs)

    if weights:
        model.load_state_dict(
            weights.get_state_dict(progress=True, map_location='cpu'), strict=True
        )

    return model
