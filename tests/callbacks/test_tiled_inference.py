# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for tiled inference callback."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader

from torchgeo.callbacks import TiledInferenceCallback
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import GridGeoSampler


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

        callback.on_predict_start(MockTrainer(), None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        assert callback.temp_dir is not None
        assert callback.temp_dir.exists()
        assert callback.crs == 'EPSG:32631'

    def test_on_predict_batch_end_saves_patches(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test on_predict_batch_end saves patches to disk."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()
        callback.crs = 'EPSG:32631'

        outputs = {
            'probabilities': torch.randn(2, 5, 64, 64),
            'bounds': torch.tensor(
                [
                    [0.0, 64.0, 1.0, 0.0, 64.0, 1.0, 0.0, 1.0, 1.0],
                    [64.0, 128.0, 1.0, 0.0, 64.0, 1.0, 0.0, 1.0, 1.0],
                ]
            ),
            'transform': torch.randn(2, 6),
        }
        batch: dict[str, torch.Tensor] = {}

        callback.on_predict_batch_end(None, None, outputs, batch, 0)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        assert len(callback.patch_metadata) == 2
        assert (callback.temp_dir / 'patch_000000.tif').exists()
        assert (callback.temp_dir / 'patch_000001.tif').exists()

        meta = callback.patch_metadata[0]
        assert 'patch_id' in meta
        assert 'file' in meta
        assert 'geo_bbox' in meta
        assert 'transform' in meta
        assert meta['geo_bbox'] == (0.0, 0.0, 64.0, 64.0)

    def test_on_predict_batch_end_missing_bounds_raises(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test error when bounds is missing."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()

        outputs = {
            'probabilities': torch.randn(2, 5, 64, 64),
            'bounds': None,
            'transform': torch.randn(2, 6),
        }

        with pytest.raises(ValueError, match=r'bounds.*is None'):
            callback.on_predict_batch_end(None, None, outputs, {}, 0)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_on_predict_batch_end_missing_transform_raises(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test error when transform is missing."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()

        outputs = {
            'probabilities': torch.randn(2, 5, 64, 64),
            'bounds': torch.randn(2, 9),
            'transform': None,
        }

        with pytest.raises(ValueError, match=r'transform.*is None'):
            callback.on_predict_batch_end(None, None, outputs, {}, 0)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_on_predict_epoch_end_cleanup(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test on_predict_epoch_end cleans up temp files."""
        callback.temp_dir = tmp_path / '.tmp_test'
        callback.temp_dir.mkdir()
        callback.crs = 'EPSG:32631'
        callback.num_classes = 5

        patch_file = callback.temp_dir / 'patch_000000.tif'
        logits = torch.randn(5, 64, 64)
        one_hot = (
            torch.nn.functional.one_hot(logits.argmax(dim=0).long(), num_classes=5)
            .permute(2, 0, 1)
            .to(torch.uint8)
            .numpy()
        )
        with rasterio.open(
            patch_file,
            'w',
            driver='GTiff',
            height=64,
            width=64,
            count=5,
            dtype='uint8',
            compress='lzw',
            tiled=True,
            transform=rasterio.transform.from_bounds(0, 0, 64, 64, 64, 64),
            crs='EPSG:32631',
        ) as dst:
            dst.write(one_hot)

        callback.patch_metadata = [
            {
                'patch_id': 0,
                'file': patch_file,
                'geo_bbox': (0.0, 0.0, 64.0, 64.0),
                'transform': [1.0, 0, 0, 0, -1.0, 100],
            }
        ]

        callback.on_predict_epoch_end(None, None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        assert not callback.temp_dir.exists()
        assert callback.output_path.exists()

    def test_on_predict_epoch_end_no_patches_raises(
        self, callback: TiledInferenceCallback
    ) -> None:
        """Test error when no patches collected."""
        with pytest.raises(ValueError, match='No patches to merge'):
            callback.on_predict_epoch_end(None, None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_on_predict_start_nested_dataset(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test on_predict_start with nested dataset (e.g., IntersectionDataset)."""
        import pandas as pd

        class MockIndex:
            bounds = pd.DataFrame(
                {'minx': [0.0], 'miny': [0.0], 'maxx': [100.0], 'maxy': [100.0]}
            )

        class MockInnerDataset:
            crs = 'EPSG:32631'
            res = (1.0, 1.0)
            index = MockIndex()

        class MockOuterDataset:
            dataset = MockInnerDataset()

        class MockDatamodule:
            predict_dataset = MockOuterDataset()

        class MockTrainer:
            datamodule = MockDatamodule()

        callback.on_predict_start(MockTrainer(), None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        assert callback.crs == 'EPSG:32631'
        assert callback.dataset_res == (1.0, 1.0)
        assert callback.dataset_bounds == (0.0, 0.0, 100.0, 100.0)

    def test_on_predict_start_disables_prediction_storage(
        self, callback: TiledInferenceCallback, tmp_path: Path
    ) -> None:
        """Test that prediction storage is disabled to prevent memory leak."""

        class MockPredictLoop:
            return_predictions = True

        class MockDataset:
            crs = 'EPSG:32631'

        class MockDatamodule:
            predict_dataset = MockDataset()

        class MockTrainer:
            predict_loop = MockPredictLoop()
            datamodule = MockDatamodule()

        callback.on_predict_start(MockTrainer(), None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        assert MockTrainer.predict_loop.return_predictions is False


class _TinySegTask(LightningModule):
    """Minimal segmentation task for integration tests."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.model = torch.nn.Conv2d(in_channels, num_classes, 1)

    def predict_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        x = batch['image']
        y_hat = self.model(x).softmax(dim=1)
        return {
            'probabilities': y_hat,
            'bounds': batch.get('bounds'),
            'transform': batch.get('transform'),
        }


class _PredictDataModule(LightningDataModule):
    """Minimal data module wiring a RasterDataset + GridGeoSampler for prediction."""

    def __init__(
        self, dataset_dir: Path, patch_size: int, stride: int, batch_size: int = 4
    ) -> None:
        super().__init__()
        self.predict_dataset = RasterDataset(paths=dataset_dir)
        self._sampler = GridGeoSampler(
            self.predict_dataset, size=patch_size, stride=stride
        )
        self._batch_size = batch_size

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset, batch_size=self._batch_size, sampler=self._sampler
        )


class TestTiledInferenceIntegration:
    """End-to-end integration tests for TiledInferenceCallback."""

    @pytest.fixture
    def synthetic_raster(self, tmp_path: Path) -> Path:
        """128x128, 3-band float32 GeoTIFF in EPSG:32631 at 1 m/px resolution."""
        scene_dir = tmp_path / 'scene'
        scene_dir.mkdir()
        raster_path = scene_dir / 'scene.tif'
        transform = rasterio.transform.from_origin(500000.0, 5000128.0, 1.0, 1.0)
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, (3, 128, 128)).astype(np.float32)
        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=128,
            width=128,
            count=3,
            dtype='float32',
            crs='EPSG:32631',
            transform=transform,
        ) as dst:
            dst.write(data)
        return scene_dir

    def test_produces_valid_geotiff(
        self, synthetic_raster: Path, tmp_path: Path
    ) -> None:
        """Full predict loop writes a valid, georeferenced uint8 GeoTIFF."""
        patch_size = 64
        overlap = 16
        delta = 8
        stride = patch_size - 2 * overlap  # 32 px

        output_path = tmp_path / 'out' / 'prediction.tif'
        output_path.parent.mkdir()

        callback = TiledInferenceCallback(
            output_path=output_path, overlap=overlap, delta=delta, blend_method='cosine'
        )
        task = _TinySegTask(in_channels=3, num_classes=2)
        dm = _PredictDataModule(synthetic_raster, patch_size=patch_size, stride=stride)

        trainer = Trainer(
            callbacks=[callback],
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.predict(task, datamodule=dm)

        assert output_path.exists(), 'output GeoTIFF not created'

        temp_dir = output_path.parent / f'.tmp_{output_path.stem}'
        assert not temp_dir.exists(), 'temp dir not cleaned up after predict'

        with rasterio.open(output_path) as src:
            assert src.crs.to_epsg() == 32631
            out_data = src.read(1)
            assert out_data.dtype == np.uint8
            assert set(np.unique(out_data)).issubset({0, 1})
            b = src.bounds
            # output should lie within the input scene bounds (500000..500128, 5000000..5000128)
            assert b.left >= 500000.0 - 1.0
            assert b.right <= 500128.0 + 1.0
            assert b.bottom >= 5000000.0 - 1.0
            assert b.top <= 5000128.0 + 1.0

    def test_return_predictions_disabled(
        self, synthetic_raster: Path, tmp_path: Path
    ) -> None:
        """on_predict_start disables Lightning's prediction storage."""
        patch_size = 64
        stride = 32

        callback = TiledInferenceCallback(
            output_path=tmp_path / 'pred.tif', overlap=16, delta=8
        )
        task = _TinySegTask()
        dm = _PredictDataModule(synthetic_raster, patch_size=patch_size, stride=stride)

        trainer = Trainer(
            callbacks=[callback],
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.predict(task, datamodule=dm)

        assert trainer.predict_loop.return_predictions is False

    def test_crs_propagated_from_dataset(
        self, synthetic_raster: Path, tmp_path: Path
    ) -> None:
        """CRS read from predict_dataset flows through to the output file."""
        callback = TiledInferenceCallback(
            output_path=tmp_path / 'pred_crs.tif', overlap=16, delta=8
        )
        task = _TinySegTask()
        dm = _PredictDataModule(synthetic_raster, patch_size=64, stride=32)

        trainer = Trainer(
            callbacks=[callback],
            accelerator='cpu',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.predict(task, datamodule=dm)

        assert callback.crs is not None
        with rasterio.open(tmp_path / 'pred_crs.tif') as src:
            assert src.crs.to_epsg() == 32631
