# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
import torch.distributed
from pytest import MonkeyPatch
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn

from torchgeo.tasks import ssl_utils
from torchgeo.tasks.ssl_utils import (
    LARS,
    MAEDecoderTIMM,
    MaskedVisionTransformerTIMM,
    NTXentLoss,
    ProjectionHead,
    cosine_schedule,
    deactivate_requires_grad,
    eye_rank,
    gather,
    get_at_index,
    normalize_mean_var,
    patchify,
    random_token_mask,
    rank,
    repeat_token,
    set_at_index,
    update_momentum,
    world_size,
)


def fake_all_gather(output: list[Tensor], input: Tensor) -> None:
    for i in range(len(output)):
        output[i].copy_(input)


class TestDistUtils:
    def test_single_process(self) -> None:
        assert rank() == 0
        assert world_size() == 1

    def test_gather(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
        monkeypatch.setattr(torch.distributed, 'all_gather', fake_all_gather)
        monkeypatch.setattr(torch.distributed, 'all_reduce', lambda x: None)

        x = torch.rand(4, 2, requires_grad=True)
        out = gather(x)
        assert len(out) == 2
        torch.cat(out, dim=0).sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

        y = torch.rand(4, 2)
        gathered = ssl_utils.concat_all_gather(y)
        assert gathered.shape == (8, 2)

    def test_eye_rank(self, monkeypatch: MonkeyPatch) -> None:
        assert torch.equal(eye_rank(3), torch.eye(3, dtype=torch.bool))

        monkeypatch.setattr(ssl_utils, 'rank', lambda: 1)
        monkeypatch.setattr(ssl_utils, 'world_size', lambda: 2)
        mask = eye_rank(3)
        assert mask.shape == (3, 6)
        assert torch.equal(mask[:, 3:], torch.eye(3, dtype=torch.bool))
        assert not mask[:, :3].any()


class TestCosineSchedule:
    def test_start_middle_end(self) -> None:
        assert cosine_schedule(0, 10, 0.99, 1.0) == pytest.approx(0.99)
        mid = cosine_schedule(5, 11, 0.0, 1.0)
        assert mid == pytest.approx(0.5)
        assert cosine_schedule(9, 10, 0.99, 1.0) == pytest.approx(1.0)
        assert cosine_schedule(20, 10, 0.99, 1.0) == pytest.approx(1.0)

    def test_single_step(self) -> None:
        assert cosine_schedule(0, 1, 0.99, 1.0) == pytest.approx(1.0)


class TestMomentumUtils:
    def test_deactivate_requires_grad(self) -> None:
        model = nn.Linear(2, 2)
        deactivate_requires_grad(model)
        assert all(not p.requires_grad for p in model.parameters())

    def test_update_momentum(self) -> None:
        model = nn.Linear(2, 2)
        model_ema = nn.Linear(2, 2)
        expected = [
            0.9 * p_ema.detach() + 0.1 * p.detach()
            for p_ema, p in zip(model_ema.parameters(), model.parameters(), strict=True)
        ]
        update_momentum(model, model_ema, m=0.9)
        for p_ema, exp in zip(model_ema.parameters(), expected, strict=True):
            assert torch.allclose(p_ema, exp)


class TestProjectionHead:
    def test_batch_norm(self) -> None:
        head = ProjectionHead(8, 16, 4, num_layers=3, batch_norm=True)
        types = [type(m) for m in head.layers]
        assert types == [
            nn.Linear,
            nn.BatchNorm1d,
            nn.ReLU,
            nn.Linear,
            nn.BatchNorm1d,
            nn.ReLU,
            nn.Linear,
            nn.BatchNorm1d,
        ]
        assert all(m.bias is None for m in head.layers if isinstance(m, nn.Linear))
        assert head(torch.rand(2, 8)).shape == (2, 4)

    def test_no_batch_norm(self) -> None:
        head = ProjectionHead(8, 16, 4, num_layers=2, batch_norm=False)
        types = [type(m) for m in head.layers]
        assert types == [nn.Linear, nn.ReLU, nn.Linear]
        assert all(m.bias is not None for m in head.layers if isinstance(m, nn.Linear))
        assert head(torch.rand(2, 8)).shape == (2, 4)


class TestNTXentLoss:
    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValueError, match='Illegal temperature'):
            NTXentLoss(temperature=0.0)

    def test_distributed_unavailable(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(torch.distributed, 'is_available', lambda: False)
        with pytest.raises(ValueError, match='torch.distributed is not available'):
            NTXentLoss(gather_distributed=True)

    def test_in_batch_negatives(self) -> None:
        loss_fn = NTXentLoss(temperature=0.07)
        out0 = torch.rand(4, 8, requires_grad=True)
        out1 = torch.rand(4, 8, requires_grad=True)
        loss = loss_fn(out0, out1)
        assert loss.isfinite()
        loss.backward()

    def test_memory_bank(self) -> None:
        loss_fn = NTXentLoss(temperature=0.07, memory_bank_size=(6, 8))
        out0 = torch.rand(4, 8, requires_grad=True)
        out1 = torch.rand(4, 8)

        # Bank is not updated during evaluation
        loss = loss_fn(out0.detach(), out1)
        assert loss.isfinite()
        assert int(loss_fn.bank_ptr) == 0

        # Bank is updated during training
        loss = loss_fn(out0, out1)
        assert loss.isfinite()
        assert int(loss_fn.bank_ptr) == 4

        # Bank pointer wraps around when full
        loss = loss_fn(out0, out1)
        assert int(loss_fn.bank_ptr) == 0

    def test_gather_distributed(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(ssl_utils, 'world_size', lambda: 2)
        monkeypatch.setattr(ssl_utils, 'gather', lambda x: (x, x))
        loss_fn = NTXentLoss(temperature=0.07, gather_distributed=True)
        out0 = torch.rand(4, 8)
        out1 = torch.rand(4, 8)
        loss = loss_fn(out0, out1)
        assert loss.isfinite()

    def test_gather_distributed_memory_bank(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(ssl_utils, 'world_size', lambda: 2)
        monkeypatch.setattr(ssl_utils, 'concat_all_gather', lambda x: torch.cat([x, x]))
        loss_fn = NTXentLoss(
            temperature=0.07, memory_bank_size=(16, 8), gather_distributed=True
        )
        out0 = torch.rand(4, 8, requires_grad=True)
        out1 = torch.rand(4, 8)
        loss = loss_fn(out0, out1)
        assert loss.isfinite()
        assert int(loss_fn.bank_ptr) == 8


class TestLARS:
    def test_invalid_hyperparams(self) -> None:
        params = nn.Linear(2, 2).parameters()
        with pytest.raises(ValueError, match='Invalid learning rate'):
            LARS(params, lr=-1)
        with pytest.raises(ValueError, match='Invalid momentum value'):
            LARS(params, lr=1, momentum=-1)
        with pytest.raises(ValueError, match='Invalid weight_decay value'):
            LARS(params, lr=1, weight_decay=-1)

    def test_step(self) -> None:
        w = nn.Parameter(torch.ones(2, 2))
        unused = nn.Parameter(torch.ones(2, 2))
        zero = nn.Parameter(torch.zeros(2, 2))
        optimizer = LARS([w, unused, zero], lr=0.1, momentum=0.9, weight_decay=1e-4)

        def closure() -> float:
            optimizer.zero_grad()
            loss = (w**2).sum() + (zero**2).sum() + zero.sum()
            loss.backward()
            return loss.item()

        start = w.detach().clone()
        # Two steps to cover both momentum buffer branches
        optimizer.step(closure)
        loss = optimizer.step(closure)
        assert loss is not None
        assert not torch.allclose(w, start)
        # Parameter without gradient is skipped
        assert torch.allclose(unused, torch.ones(2, 2))


class TestTokenUtils:
    def test_repeat_token(self) -> None:
        token = torch.rand(1, 1, 3)
        out = repeat_token(token, (2, 5))
        assert out.shape == (2, 5, 3)
        assert torch.allclose(out[1, 4], token[0, 0])

    def test_get_set_at_index(self) -> None:
        tokens = torch.rand(2, 5, 3)
        index = torch.tensor([[0, 2], [1, 4]])
        value = torch.rand(2, 2, 3)
        out = set_at_index(tokens, index, value)
        assert torch.allclose(get_at_index(out, index), value)

    def test_patchify(self) -> None:
        images = torch.rand(2, 3, 4, 4)
        patches = patchify(images, 2)
        assert patches.shape == (2, 4, 12)
        expected = images[0, :, :2, :2].permute(1, 2, 0).reshape(-1)
        assert torch.allclose(patches[0, 0], expected)

    def test_random_token_mask(self) -> None:
        idx_keep, idx_mask = random_token_mask((2, 17), mask_ratio=0.75)
        assert idx_keep.shape == (2, 5)
        assert idx_mask.shape == (2, 12)
        for row in idx_keep:
            assert 0 in row
        union = torch.cat([idx_keep, idx_mask], dim=1).sort(dim=1).values
        assert torch.equal(union, torch.arange(17).expand(2, -1))

    def test_normalize_mean_var(self) -> None:
        x = torch.rand(2, 5, 16)
        out = normalize_mean_var(x)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
        assert torch.allclose(out.var(dim=-1), torch.ones(2, 5), atol=1e-3)


class TestMaskedVisionTransformerTIMM:
    def vit(self, **kwargs: bool | int | str) -> VisionTransformer:
        return VisionTransformer(
            img_size=32,
            patch_size=8,
            embed_dim=32,
            depth=1,
            num_heads=2,
            num_classes=0,
            **kwargs,  # type: ignore[invalid-argument-type]
        )

    def test_encode(self) -> None:
        model = MaskedVisionTransformerTIMM(vit=self.vit())
        assert model.sequence_length == 17
        assert not model.vit.pos_embed.requires_grad
        images = torch.rand(2, 3, 32, 32)
        tokens = model.encode(images)
        assert tokens.shape == (2, 17, 32)
        idx_keep = torch.arange(5).expand(2, -1)
        tokens = model.encode(images, idx_keep=idx_keep)
        assert tokens.shape == (2, 5, 32)

    def test_no_class_token(self) -> None:
        model = MaskedVisionTransformerTIMM(
            vit=self.vit(class_token=False, global_pool='avg')
        )
        assert model.sequence_length == 16
        tokens = model.encode(torch.rand(2, 3, 32, 32))
        assert tokens.shape == (2, 16, 32)

    def test_unsupported_vit(self) -> None:
        with pytest.raises(AssertionError, match='dynamic image size'):
            MaskedVisionTransformerTIMM(vit=self.vit(dynamic_img_size=True))
        with pytest.raises(AssertionError, match='no_embed_class'):
            MaskedVisionTransformerTIMM(vit=self.vit(no_embed_class=True))
        with pytest.raises(AssertionError, match='register tokens'):
            MaskedVisionTransformerTIMM(vit=self.vit(reg_tokens=1))


class TestMAEDecoderTIMM:
    def test_decoder(self) -> None:
        decoder = MAEDecoderTIMM(
            num_patches=16,
            patch_size=8,
            in_chans=3,
            embed_dim=32,
            decoder_embed_dim=16,
            decoder_depth=1,
            decoder_num_heads=2,
        )
        assert decoder.mask_token.shape == (1, 1, 16)
        x = torch.rand(2, 17, 32)
        x = decoder.embed(x)
        assert x.shape == (2, 17, 16)
        x = decoder.decode(x)
        assert x.shape == (2, 17, 16)
        x = decoder.predict(x)
        assert x.shape == (2, 17, 8 * 8 * 3)
