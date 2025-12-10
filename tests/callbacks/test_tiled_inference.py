# Copyright (c) TorchGeo Contributors
# All rights reserved.
# Licensed under the MIT License.

"""Tests for tiled inference callback."""

from pathlib import Path

import pytest
import torch

from torchgeo.callbacks import TiledInferenceCallback


class TestTiledInferenceCallback:
    """Tests for TiledInferenceCallback."""

    @pytest.fixture
    def callback(self, tmp_path: Path) -> TiledInferenceCallback:
        """Create callback instance.

        Args:
            tmp_path: Temporary directory path.

        Returns:
            TiledInferenceCallback instance.
        """
        return TiledInferenceCallback(
            output_path=tmp_path / 'output.tif', overlap=32, delta=8
        )

    def test_init(self, callback: TiledInferenceCallback) -> None:
        """Test callback initialization."""
        assert callback.overlap == 32
        assert callback.delta == 8
        assert callback.blend_method == 'cosine'
        assert callback.chunk_size == 4096

    def test_on_predict_start(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test on_predict_start creates temp directory."""

        class MockDataset:
            crs = 'EPSG:32631'

        class MockDatamodule:
            predict_dataset = MockDataset()

        class MockTrainer:
            datamodule = MockDatamodule()

        callback.on_predict_start(MockTrainer(), None)

        assert callback.temp_dir is not None
        assert callback.temp_dir.exists()
        assert callback.crs == 'EPSG:32631'

    def test_on_predict_batch_end_saves_patches(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test on_predict_batch_end saves patches to disk."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()

        outputs = {
            'logits': torch.randn(2, 5, 64, 64),
            'bounds': torch.tensor(
                [[0, 64, 1, 0, 64, 1, 0, 1, 1], [64, 128, 1, 0, 64, 1, 0, 1, 1]]
            ),
            'transform': torch.randn(2, 6),
        }
        batch: dict[str, torch.Tensor] = {}

        callback.on_predict_batch_end(None, None, outputs, batch, 0)

        assert len(callback.patch_metadata) == 2
        assert (callback.temp_dir / 'patch_000000.pt').exists()
        assert (callback.temp_dir / 'patch_000001.pt').exists()

        meta = callback.patch_metadata[0]
        assert 'patch_id' in meta
        assert 'file' in meta
        assert 'bbox' in meta
        assert 'transform' in meta
        assert meta['bbox'] == (0, 0, 64, 64)

    def test_on_predict_batch_end_missing_bounds_raises(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test error when bounds is missing."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()

        outputs = {
            'logits': torch.randn(2, 5, 64, 64),
            'bounds': None,
            'transform': torch.randn(2, 6),
        }

        with pytest.raises(ValueError, match=r'bounds.*is None'):
            callback.on_predict_batch_end(None, None, outputs, {}, 0)

    def test_on_predict_batch_end_missing_transform_raises(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test error when transform is missing."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()

        outputs = {
            'logits': torch.randn(2, 5, 64, 64),
            'bounds': torch.randn(2, 9),
            'transform': None,
        }

        with pytest.raises(ValueError, match=r'transform.*is None'):
            callback.on_predict_batch_end(None, None, outputs, {}, 0)

    def test_on_predict_epoch_end_cleanup(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test on_predict_epoch_end cleans up temp files."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()
        callback.crs = 'EPSG:32631'
        callback.num_classes = 5

        patch_file = callback.temp_dir / 'patch_000000.pt'
        torch.save(
            {
                'logits': torch.randn(5, 64, 64),
                'bounds': torch.tensor([0, 64, 1, 0, 64, 1, 0, 1, 1]),
                'transform': torch.tensor([1.0, 0, 0, 0, -1.0, 100]),
            },
            patch_file,
        )

        callback.patch_metadata = [
            {
                'patch_id': 0,
                'file': patch_file,
                'bbox': (0, 0, 64, 64),
                'transform': torch.tensor([1.0, 0, 0, 0, -1.0, 100]),
            }
        ]

        callback.on_predict_epoch_end(None, None)

        assert not callback.temp_dir.exists()
        assert callback.output_path.exists()

    def test_on_predict_epoch_end_no_patches_raises(
        self, callback: TiledInferenceCallback
    ) -> None:
        """Test error when no patches collected."""
        with pytest.raises(ValueError, match='No patches to merge'):
            callback.on_predict_epoch_end(None, None)
