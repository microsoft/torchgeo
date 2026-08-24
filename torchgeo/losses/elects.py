# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Loss function for early classification of time series."""

import torch
from torch import Tensor
from torch.nn import NLLLoss
from torch.nn.modules import Module


def _decision_probability(probability_stopping: Tensor) -> Tensor:
    """Compute the probability of making a decision at each time step."""
    remaining = torch.cumprod(1 - probability_stopping[:, :-1], dim=1)
    budget = torch.cat([torch.ones_like(probability_stopping[:, :1]), remaining], dim=1)
    return torch.cat(
        [probability_stopping[:, :-1] * budget[:, :-1], budget[:, -1:]], dim=1
    )


class EarlyRewardLoss(Module):
    """ELECTS loss for early classification of time series.

    This loss is defined in `'End-to-end learned early classification of time
    series for in-season crop type mapping'
    <https://doi.org/10.1016/j.isprsjprs.2022.12.016>`_. It balances
    classification accuracy against making correct predictions early in a sequence.

    .. versionadded:: 0.11
    """

    def __init__(
        self, alpha: float = 0.5, epsilon: float = 10, weight: Tensor | None = None
    ) -> None:
        """Initialize a new EarlyRewardLoss instance.

        Args:
            alpha: Trade-off between classification accuracy and earliness. Must be
                between 0 and 1.
            epsilon: Additive smoothing applied to the decision probabilities. Must
                be greater than or equal to 0.
            weight: Manual rescaling weight for each class.

        Raises:
            ValueError: If ``alpha`` or ``epsilon`` is outside its valid range.
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f'Invalid alpha value: {alpha}')
        if epsilon < 0:
            raise ValueError(f'Invalid epsilon value: {epsilon}')

        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.nll_loss = NLLLoss(weight=weight, reduction='none')

    def forward(
        self,
        log_probs: Tensor,
        probability_stopping: Tensor,
        target: Tensor,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute the ELECTS early classification loss.

        Args:
            log_probs: Log class probabilities with shape B x T x C x ... .
            probability_stopping: Probability of stopping with shape B x T x ... .
                The final value is ignored because a decision is always made at the
                final time step.
            target: Class indices with shape B x ... . Targets already repeated over
                time with shape B x T x ... are also supported.
            return_stats: Return the loss components and decision probabilities.

        Returns:
            The scalar loss, or the loss and detached statistics when
            ``return_stats`` is true.

        Raises:
            ValueError: If any input has an incompatible shape.
        """
        if log_probs.ndim < 3:
            raise ValueError(
                'log_probs must have shape B x T x C x ..., '
                f'but found {tuple(log_probs.shape)}'
            )

        batch_size, sequence_length, num_classes, *spatial_shape = log_probs.shape
        expected_stopping_shape = (batch_size, sequence_length, *spatial_shape)
        if probability_stopping.shape != expected_stopping_shape:
            raise ValueError(
                'probability_stopping must have shape '
                f'{expected_stopping_shape}, but found '
                f'{tuple(probability_stopping.shape)}'
            )

        expected_target_shape = (batch_size, *spatial_shape)
        temporal_target_shape = (batch_size, sequence_length, *spatial_shape)
        if target.shape == expected_target_shape:
            target = target.unsqueeze(dim=1).expand(temporal_target_shape)
        elif target.shape != temporal_target_shape:
            raise ValueError(
                f'target must have shape {expected_target_shape} or '
                f'{temporal_target_shape}, but found {tuple(target.shape)}'
            )

        decision_probability = _decision_probability(probability_stopping)
        decision_probability = decision_probability + self.epsilon / sequence_length

        time = torch.arange(
            sequence_length, device=log_probs.device, dtype=log_probs.dtype
        )
        time = time.reshape(1, sequence_length, *([1] * len(spatial_shape)))
        correct_probability = (
            log_probs.gather(dim=2, index=target.unsqueeze(dim=2)).squeeze(dim=2).exp()
        )
        earliness_reward = (
            (decision_probability * correct_probability * (1 - time / sequence_length))
            .sum(dim=1)
            .mean()
        )

        classification_loss = self.nll_loss(
            log_probs.reshape(
                batch_size * sequence_length, num_classes, *spatial_shape
            ),
            target.reshape(batch_size * sequence_length, *spatial_shape),
        ).reshape(batch_size, sequence_length, *spatial_shape)
        classification_loss = (
            (classification_loss * decision_probability).sum(dim=1).mean()
        )

        loss = self.alpha * classification_loss - (1 - self.alpha) * earliness_reward
        if return_stats:
            stats = {
                'classification_loss': classification_loss.detach(),
                'earliness_reward': earliness_reward.detach(),
                'probability_making_decision': decision_probability.detach(),
            }
            return loss, stats
        return loss
