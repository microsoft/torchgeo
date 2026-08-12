# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.losses import EarlyRewardLoss
from torchgeo.losses.elects import _decision_probability


class TestEarlyRewardLoss:
    @pytest.fixture
    def inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.tensor(
            [
                [[2.0, 1.0], [1.0, 2.0], [2.0, 1.0]],
                [[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]],
            ],
            requires_grad=True,
        )
        log_probs = logits.log_softmax(dim=-1)
        probability_stopping = torch.tensor([[0.2, 0.5, 0.9], [0.4, 0.5, 0.9]])
        target = torch.tensor([[0, 0, 0], [1, 1, 1]])
        return log_probs, probability_stopping, target

    def test_decision_probability(self) -> None:
        probability_stopping = torch.tensor([[0.2, 0.5, 0.9]])
        actual = _decision_probability(probability_stopping)
        expected = torch.tensor([[0.2, 0.4, 0.4]])
        assert torch.allclose(actual, expected)

    def test_single_time_step(self) -> None:
        actual = _decision_probability(torch.tensor([[0.5], [0.1]]))
        assert torch.equal(actual, torch.ones(2, 1))

    def test_forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        log_probs, probability_stopping, target = inputs
        loss = EarlyRewardLoss(epsilon=0)(log_probs, probability_stopping, target)
        assert loss.ndim == 0
        loss.backward()
        assert log_probs.grad_fn is not None

    def test_return_stats(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        log_probs, probability_stopping, target = inputs
        output = EarlyRewardLoss()(log_probs, probability_stopping, target, True)
        assert isinstance(output, tuple)
        loss, stats = output
        assert loss.ndim == 0
        assert set(stats) == {
            'classification_loss',
            'earliness_reward',
            'probability_making_decision',
        }
        assert stats['probability_making_decision'].shape == target.shape
        assert all(not value.requires_grad for value in stats.values())

    def test_weight(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        log_probs, probability_stopping, target = inputs
        loss = EarlyRewardLoss(weight=torch.tensor([1.0, 2.0]))(
            log_probs, probability_stopping, target
        )
        assert torch.isfinite(loss)

    @pytest.mark.parametrize('alpha', [-0.1, 1.1])
    def test_invalid_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match='Invalid alpha value'):
            EarlyRewardLoss(alpha=alpha)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError, match='Invalid epsilon value'):
            EarlyRewardLoss(epsilon=-1)
