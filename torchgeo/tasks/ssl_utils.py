# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Building blocks for self-supervised learning (SSL) tasks.

This module contains minimal implementations of the components used by
TorchGeo's SimCLR, MoCo, and MAE trainers, adapted from `Lightly
<https://github.com/lightly-ai/lightly>`__ (MIT License, Copyright (c) 2020
Lightly AG). The LARS optimizer is in turn adapted from `PyTorch Lightning
Bolts <https://github.com/Lightning-Universe/lightning-bolts>`__
(Apache-2.0 License).

.. versionadded:: 0.10
"""

import itertools
import math
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from numpy.typing import NDArray
from timm.models.vision_transformer import Block, VisionTransformer
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer


def rank() -> int:
    """Return the rank of the current process.

    Returns:
        The rank of the current process, or 0 if torch.distributed is not initialized.
    """
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    """Return the number of distributed processes.

    Returns:
        The world size, or 1 if torch.distributed is not initialized.
    """
    return dist.get_world_size() if dist.is_initialized() else 1


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all processes, supporting backpropagation.

    Adapted from the Solo-learn project:
    https://github.com/vturrisi/solo-learn
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor) -> tuple[Tensor, ...]:
        """Gather *input* from all processes.

        Args:
            ctx: Autograd context.
            input: Tensor to gather.

        Returns:
            A tuple with one tensor per process.
        """
        output = [torch.empty_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> Tensor:
        """Reduce gradients from all processes.

        Args:
            ctx: Autograd context.
            grads: Gradients with respect to each forward output.

        Returns:
            The gradient with respect to *input*.
        """
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather(input: Tensor) -> tuple[Tensor, ...]:
    """Gather a tensor from all processes with support for backpropagation.

    Args:
        input: Tensor to gather.

    Returns:
        A tuple with one tensor per process.
    """
    return cast(tuple[Tensor, ...], GatherLayer.apply(input))


@torch.no_grad()
def concat_all_gather(x: Tensor) -> Tensor:
    """Gather a tensor from all processes and concatenate, without gradients.

    Args:
        x: Tensor to gather.

    Returns:
        The gathered tensors concatenated along the batch dimension.
    """
    output = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(output, x)
    return torch.cat(output, dim=0)


def eye_rank(n: int, device: torch.device | None = None) -> Tensor:
    """Return an (n, n * world_size) boolean matrix with a rank-offset diagonal.

    The diagonal for the block corresponding to the rank of the current process
    is set to True. Equivalent to :func:`torch.eye` if the world size is 1.

    Args:
        n: Size of the square matrix on a single process.
        device: Device on which the matrix should be created.

    Returns:
        A boolean tensor with the diagonal for the current rank set to True.
    """
    rows = torch.arange(n, device=device, dtype=torch.long)
    cols = rows + rank() * n
    diag_mask = torch.zeros((n, n * world_size()), dtype=torch.bool, device=device)
    diag_mask[(rows, cols)] = True
    return diag_mask


def cosine_schedule(
    step: int, max_steps: int, start_value: float, end_value: float
) -> float:
    """Use cosine decay to gradually modify *start_value* to reach *end_value*.

    Args:
        step: Current step number.
        max_steps: Total number of steps.
        start_value: Starting value.
        end_value: Target value.

    Returns:
        Cosine decay value for the current step.
    """
    if max_steps <= 1 or step >= max_steps - 1:
        # Lightning also updates the schedule for the epoch after the last
        # training epoch, so anything past the end returns the end value.
        return end_value
    return (
        end_value
        - (end_value - start_value)
        * (math.cos(math.pi * step / (max_steps - 1)) + 1)
        / 2
    )


def deactivate_requires_grad(model: nn.Module) -> None:
    """Deactivate the requires_grad flag for all parameters of a model.

    Args:
        model: Model whose parameters are frozen.
    """
    for param in model.parameters():
        param.requires_grad = False


@torch.no_grad()
def update_momentum(model: nn.Module, model_ema: nn.Module, m: float) -> None:
    """Update the parameters of *model_ema* with an exponential moving average.

    Args:
        model: The current model.
        model_ema: The model with exponential moving average (EMA) parameters.
        m: The momentum factor, between 0 and 1.
    """
    params = list(model.parameters())
    ema_params = list(model_ema.parameters())
    torch._foreach_mul_(ema_params, m)
    torch._foreach_add_(ema_params, params, alpha=1.0 - m)


class ProjectionHead(nn.Module):
    """MLP projection head used by SimCLR and MoCo.

    Hidden layers are followed by an optional batch norm layer and a ReLU
    non-linearity. The final layer is followed by an optional batch norm layer
    and no non-linearity. Linear layers directly followed by batch norm have
    no bias term.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        batch_norm: bool = True,
    ) -> None:
        """Initialize a new ProjectionHead instance.

        Args:
            input_dim: Number of input dimensions.
            hidden_dim: Number of hidden dimensions.
            output_dim: Number of output dimensions.
            num_layers: Total number of linear layers.
            batch_norm: Whether or not to use batch norm layers.
        """
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            layers.append(nn.Linear(in_dim, out_dim, bias=not batch_norm))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the projection head.

        Args:
            x: Input mini-batch of shape (batch_size, input_dim).

        Returns:
            Projected output of shape (batch_size, output_dim).
        """
        out: Tensor = self.layers(x)
        return out


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss.

    This implementation follows the SimCLR paper. If the memory bank is
    enabled by setting *memory_bank_size* > 0, the loss behaves like the one
    described in the MoCo paper, with the memory bank entries used as
    negative samples.

    If you use this loss in your research, please cite the following papers:

    * https://arxiv.org/abs/2002.05709
    * https://arxiv.org/abs/1911.05722
    """

    bank: Tensor
    bank_ptr: Tensor

    def __init__(
        self,
        temperature: float = 0.5,
        memory_bank_size: tuple[int, int] = (0, 0),
        gather_distributed: bool = False,
    ) -> None:
        """Initialize a new NTXentLoss instance.

        Args:
            temperature: Scale logits by the inverse of the temperature.
            memory_bank_size: Size of the memory bank as a
                (num_negatives, embedding_dim) tuple. Use (0, 0) to disable
                the memory bank and use in-batch negatives instead.
            gather_distributed: If True, negatives from all GPUs are gathered
                before the loss calculation.

        Raises:
            ValueError: If abs(temperature) < 1e-8 (to prevent divide by zero)
                or if *gather_distributed* is True but torch.distributed is
                not available.
        """
        super().__init__()
        if abs(temperature) < 1e-8:
            raise ValueError(f'Illegal temperature: abs({temperature}) < 1e-8')
        if gather_distributed and not dist.is_available():
            raise ValueError(
                'gather_distributed is True but torch.distributed is not available. '
                'Please set gather_distributed=False or install a torch version with '
                'distributed support.'
            )
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.size = memory_bank_size[0]
        bank = torch.empty(0)
        if self.size > 0:
            bank = F.normalize(torch.randn(*memory_bank_size), dim=-1)
        self.register_buffer('bank', bank, persistent=False)
        self.register_buffer(
            'bank_ptr', torch.zeros(1, dtype=torch.long), persistent=False
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: Tensor) -> None:
        """Dequeue the oldest batch and add the latest one to the memory bank.

        Args:
            batch: The latest batch of keys to add to the memory bank.
        """
        if self.gather_distributed and world_size() > 1:
            batch = concat_all_gather(batch)
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size:
            self.bank[ptr:] = batch[: self.size - ptr].detach()
            self.bank_ptr.zero_()
        else:
            self.bank[ptr : ptr + batch_size] = batch.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(self, out0: Tensor, out1: Tensor) -> Tensor:
        """Forward pass of the contrastive cross entropy loss.

        Args:
            out0: Output projections of the first set of transformed images,
                with shape (batch_size, embedding_dim).
            out1: Output projections of the second set of transformed images,
                with shape (batch_size, embedding_dim).

        Returns:
            The contrastive cross entropy loss value.
        """
        device = out0.device
        batch_size = out0.shape[0]

        # Normalize the output to length 1
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        if self.size > 0:
            # Use negatives from the memory bank. The bank is only updated if
            # a backward pass follows (i.e., during training).
            negatives = self.bank.clone().detach().to(device)
            if out0.requires_grad:
                self._dequeue_and_enqueue(out1)

            # sim_pos[i] is the similarity of the i-th sample to its positive pair
            sim_pos = torch.einsum('nc,nc->n', out0, out1).unsqueeze(-1)
            # sim_neg[i, k] is the similarity of the i-th sample to the k-th negative
            sim_neg = torch.einsum('nc,kc->nk', out0, negatives)

            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        else:
            # Use other samples from the batch as negatives
            if self.gather_distributed and world_size() > 1:
                # Gather hidden representations from other processes
                out0_large = torch.cat(gather(out0), 0)
                out1_large = torch.cat(gather(out1), 0)
                diag_mask = eye_rank(batch_size, device=device)
            else:
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

            logits_00 = torch.einsum('nc,mc->nm', out0, out0_large) / self.temperature
            logits_01 = torch.einsum('nc,mc->nm', out0, out1_large) / self.temperature
            logits_10 = torch.einsum('nc,mc->nm', out1, out0_large) / self.temperature
            logits_11 = torch.einsum('nc,mc->nm', out1, out1_large) / self.temperature

            # Remove similarities between the same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            logits = torch.cat(
                [
                    torch.cat([logits_01, logits_00], dim=1),
                    torch.cat([logits_10, logits_11], dim=1),
                ],
                dim=0,
            )
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.gather_distributed:
                labels = labels + rank() * batch_size
            labels = labels.repeat(2)

        loss: Tensor = self.cross_entropy(logits, labels)
        return loss


class LARS(Optimizer):
    """Extends SGD with LARS scaling from "Large batch training of Convolutional Networks".

    Parameters with weight decay set to 0 are automatically excluded from
    layer-wise learning rate scaling, for consistency with papers such as
    SimCLR and BYOL.

    If you use this optimizer in your research, please cite the following paper:

    * https://arxiv.org/abs/1708.03888
    """

    def __init__(
        self,
        params: Any,
        lr: float,
        momentum: float = 0,
        weight_decay: float = 0,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ) -> None:
        """Initialize a new LARS instance.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Learning rate.
            momentum: Momentum factor.
            weight_decay: Weight decay (L2 penalty).
            trust_coefficient: Trust coefficient for computing the learning rate.
            eps: Epsilon added to the division denominator.

        Raises:
            ValueError: If *lr*, *momentum*, or *weight_decay* are negative.
        """
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'trust_coefficient': trust_coefficient,
            'eps': eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss returned by *closure*, if given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p)
                g_norm = torch.norm(p.grad)

                # Apply LARS scaling and weight decay. Scaling is skipped for
                # parameters with 0 weight decay.
                if weight_decay != 0 and p_norm != 0 and g_norm != 0:
                    lars_lr = p_norm / (g_norm + p_norm * weight_decay + group['eps'])
                    lars_lr *= group['trust_coefficient']
                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p *= lars_lr

                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss


def repeat_token(token: Tensor, size: tuple[int, int]) -> Tensor:
    """Repeat a token to fill a (batch_size, sequence_length) grid.

    Args:
        token: Token tensor with shape (1, 1, dim).
        size: A (batch_size, sequence_length) tuple.

    Returns:
        Tensor with shape (batch_size, sequence_length, dim) containing copies
        of the input token.
    """
    batch_size, sequence_length = size
    return token.repeat(batch_size, sequence_length, 1)


def expand_index_like(index: Tensor, tokens: Tensor) -> Tensor:
    """Expand an index along the last dimension of the input tokens.

    Args:
        index: Index tensor with shape (batch_size, idx_length).
        tokens: Token tensor with shape (batch_size, sequence_length, dim).

    Returns:
        Index tensor with shape (batch_size, idx_length, dim) where the
        original indices are repeated dim times along the last dimension.
    """
    dim = tokens.shape[-1]
    return index.unsqueeze(-1).expand(-1, -1, dim)


def get_at_index(tokens: Tensor, index: Tensor) -> Tensor:
    """Select tokens at the given indices.

    Args:
        tokens: Token tensor with shape (batch_size, sequence_length, dim).
        index: Index tensor with shape (batch_size, index_length).

    Returns:
        Token tensor with shape (batch_size, index_length, dim) containing the
        selected tokens.
    """
    index = expand_index_like(index, tokens)
    return torch.gather(tokens, 1, index)


def set_at_index(tokens: Tensor, index: Tensor, value: Tensor) -> Tensor:
    """Copy values into the input tensor at the given indices.

    Args:
        tokens: Token tensor with shape (batch_size, sequence_length, dim).
        index: Index tensor with shape (batch_size, index_length).
        value: Value tensor with shape (batch_size, index_length, dim).

    Returns:
        Token tensor with shape (batch_size, sequence_length, dim) containing
        the new values.
    """
    index = expand_index_like(index, tokens)
    return torch.scatter(tokens, 1, index, value)


def patchify(images: Tensor, patch_size: int) -> Tensor:
    """Convert a batch of input images into patches.

    Args:
        images: Image tensor with shape (batch_size, channels, height, width).
            Height and width must be equal and multiples of *patch_size*.
        patch_size: Patch size in pixels.

    Returns:
        Patch tensor with shape
        (batch_size, num_patches, channels * patch_size ** 2).
    """
    n, c, h, w = images.shape
    assert h == w and h % patch_size == 0

    patch_h = patch_w = h // patch_size
    num_patches = patch_h * patch_w
    patches = images.reshape(n, c, patch_h, patch_size, patch_w, patch_size)
    patches = torch.einsum('nchpwq->nhwpqc', patches)
    return patches.reshape(n, num_patches, patch_size**2 * c)


def random_token_mask(
    size: tuple[int, int],
    mask_ratio: float = 0.6,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Create random token masks. The class token at index 0 is never masked.

    Args:
        size: Size of the token batch for which to generate masks, as a
            (batch_size, sequence_length) tuple.
        mask_ratio: Proportion of tokens to mask.
        device: Device on which to create the index masks.

    Returns:
        An (index_keep, index_mask) tuple. *index_keep* contains the indices
        of the unmasked tokens and *index_mask* contains the indices of the
        masked tokens.
    """
    batch_size, sequence_length = size
    num_keep = int((sequence_length - 1) * (1 - mask_ratio))

    noise = torch.rand(batch_size, sequence_length, device=device)
    # Make sure that the class token is not masked
    noise[:, 0] = -1
    num_keep = max(1, num_keep + 1)

    # Get indices of tokens to keep by sorting the noise
    indices = torch.argsort(noise, dim=1)
    idx_keep = indices[:, :num_keep]
    idx_mask = indices[:, num_keep:]
    return idx_keep, idx_mask


def normalize_mean_var(x: Tensor, dim: int = -1, eps: float = 1.0e-6) -> Tensor:
    """Normalize the input tensor to zero mean and unit variance.

    Args:
        x: Input tensor.
        dim: Dimension along which to compute mean and standard deviation.
        eps: Epsilon value to avoid division by zero.

    Returns:
        The normalized tensor.
    """
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True)
    return (x - mean) / (var + eps).sqrt()


def _get_1d_sincos_pos_embed_from_positions(
    embed_dim: int, pos: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Generate 1D sine-cosine positional embeddings from positions.

    Args:
        embed_dim: Embedding dimension, must be even.
        pos: Positions to be encoded, with shape (N, M).

    Returns:
        Positional embedding with shape (N * M, embed_dim).
    """
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool
) -> NDArray[np.float32]:
    """Generate 2D sine-cosine positional embeddings.

    Code follows https://github.com/facebookresearch/mae.

    Args:
        embed_dim: Embedding dimension, must be even.
        grid_size: Height and width of the grid.
        cls_token: If True, a zero positional embedding for the class token is
            prepended.

    Returns:
        Positional embedding with shape (grid_size * grid_size, embed_dim), or
        (1 + grid_size * grid_size, embed_dim) if *cls_token* is True.
    """
    assert embed_dim % 2 == 0
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    # Use half of the dimensions to encode each of the grid axes
    emb_h = _get_1d_sincos_pos_embed_from_positions(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_positions(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _initialize_2d_sincos_pos_embed(
    pos_embed: nn.Parameter, has_class_token: bool
) -> None:
    """Initialize a positional embedding with fixed 2D sine-cosine values.

    The embedding is frozen after initialization.

    Args:
        pos_embed: Positional embedding parameter with shape
            (1, sequence_length, dim).
        has_class_token: Whether the first position corresponds to a class
            token.
    """
    _, seq_length, hidden_dim = pos_embed.shape
    grid_size = int((seq_length - int(has_class_token)) ** 0.5)
    sincos_embedding = _get_2d_sincos_pos_embed(
        embed_dim=hidden_dim, grid_size=grid_size, cls_token=has_class_token
    )
    pos_embed.data.copy_(torch.from_numpy(sincos_embedding).float().unsqueeze(0))
    pos_embed.requires_grad = False


def _init_weights(module: nn.Module) -> None:
    """Initialize linear and layer norm modules following the MAE paper.

    Args:
        module: Module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class MaskedVisionTransformerTIMM(nn.Module):
    """Masked Vision Transformer encoder for MAE, wrapping a timm ViT.

    The positional embedding is re-initialized with fixed 2D sine-cosine
    values and frozen, and the remaining weights are re-initialized following
    the MAE paper.
    """

    def __init__(self, vit: VisionTransformer) -> None:
        """Initialize a new MaskedVisionTransformerTIMM instance.

        Args:
            vit: The timm VisionTransformer to wrap. Models with dynamic image
                size, register tokens, or *no_embed_class* are not supported.
        """
        super().__init__()
        assert not vit.dynamic_img_size, 'dynamic image size is not supported'
        assert not vit.no_embed_class, 'no_embed_class is not supported'
        assert vit.reg_token is None, 'register tokens are not supported'
        self.vit = vit
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        self._initialize_weights()
        _initialize_2d_sincos_pos_embed(
            self.vit.pos_embed, has_class_token=self.vit.num_prefix_tokens > 0
        )

    @property
    def sequence_length(self) -> int:
        """Total sequence length, including any prefix tokens.

        Returns:
            The number of patch tokens plus the number of prefix tokens.
        """
        return self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens

    def encode(self, images: Tensor, idx_keep: Tensor | None = None) -> Tensor:
        """Encode input images, optionally keeping only a subset of tokens.

        Args:
            images: Image tensor with shape (batch_size, channels, height,
                width).
            idx_keep: Index tensor with shape (batch_size, num_tokens_to_keep)
                where each entry is an index of a token to keep in the
                respective batch. If set, only the indexed tokens are encoded.

        Returns:
            Encoded token tensor with shape (batch_size, sequence_length,
            embed_dim), or (batch_size, num_tokens_to_keep, embed_dim) if
            *idx_keep* is set.
        """
        tokens: Tensor = self.vit.patch_embed(images)
        if self.vit.cls_token is not None:
            cls_token = self.vit.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.vit.pos_embed
        tokens = self.vit.pos_drop(tokens)
        if idx_keep is not None:
            tokens = get_at_index(tokens, idx_keep)
        tokens = self.vit.norm_pre(tokens)
        tokens = self.vit.blocks(tokens)
        tokens = self.vit.norm(tokens)
        return tokens

    def _initialize_weights(self) -> None:
        """Initialize the weights of the wrapped ViT following the MAE paper."""
        # Initialize the patch embedding like a linear layer instead of a conv layer
        w = self.vit.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.vit.cls_token is not None:
            nn.init.normal_(self.vit.cls_token, std=0.02)
        self.apply(_init_weights)


class MAEDecoderTIMM(nn.Module):
    """Decoder for the Masked Autoencoder model.

    Decodes encoded patches and predicts pixel values for every patch.

    If you use this module in your research, please cite the following paper:

    * https://arxiv.org/abs/2111.06377
    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        in_chans: int = 3,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        """Initialize a new MAEDecoderTIMM instance.

        Args:
            num_patches: Number of patches.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            embed_dim: Embedding dimension of the encoder.
            decoder_embed_dim: Embedding dimension of the decoder.
            decoder_depth: Number of transformer blocks in the decoder.
            decoder_num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            proj_drop_rate: Dropout rate after the MLP in the transformer.
            attn_drop_rate: Dropout rate after the attention head.
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # Fixed sine-cosine positional encoding of the decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )

        nn.init.normal_(self.mask_token, std=0.02)
        _initialize_2d_sincos_pos_embed(self.decoder_pos_embed, has_class_token=True)
        self.apply(_init_weights)

    def embed(self, input: Tensor) -> Tensor:
        """Embed encoded input tokens into the decoder token dimension.

        Args:
            input: Tensor with shape (batch_size, seq_length, embed_dim)
                containing the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, decoder_embed_dim)
            containing the embedded tokens.
        """
        out: Tensor = self.decoder_embed(input)
        return out

    def decode(self, input: Tensor) -> Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input: Tensor with shape (batch_size, seq_length,
                decoder_embed_dim) containing the embedded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, decoder_embed_dim)
            containing the decoded tokens.
        """
        output: Tensor = input + self.decoder_pos_embed
        output = self.decoder_blocks(output)
        output = self.decoder_norm(output)
        return output

    def predict(self, input: Tensor) -> Tensor:
        """Predict pixel values from decoded tokens.

        Args:
            input: Tensor with shape (batch_size, seq_length,
                decoder_embed_dim) containing the decoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length,
            patch_size ** 2 * in_chans) containing the predictions for each
            token.
        """
        out: Tensor = self.decoder_pred(input)
        return out
