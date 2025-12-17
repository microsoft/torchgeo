# Copyright (c) TorchGeo Contributors
# All rights reserved.
# Licensed under the MIT License.

"""Tiled inference callback for semantic segmentation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class TiledInferenceCallback(Callback):
    """Callback for tiled inference with weighted blending.

    Saves patch predictions to temporary files during prediction,
    then merges with weighted blending and writes GeoTIFF output.

    Example::

        from lightning import Trainer
        from torchgeo.callbacks import TiledInferenceCallback

        callback = TiledInferenceCallback(
            output_path='predictions.tif',
            overlap=32,
            delta=8,
        )
        trainer = Trainer(callbacks=[callback])
        trainer.predict(task, datamodule)

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        output_path: str | Path,
        overlap: int = 32,
        delta: int = 8,
        blend_method: str = 'cosine',  # TODO: Get rid of linear
        chunk_size: int = 4096,
        cog_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize callback.

        Args:
            output_path: Path to save output GeoTIFF.
            overlap: Overlap in pixels on each side of patch.
            delta: Pixels to crop from patch edges before blending.
            blend_method: Blending method ('cosine' or 'linear').
            chunk_size: Chunk size for output processing (memory vs speed).
            cog_config: Optional Cloud-Optimized GeoTIFF configuration.
        """
        super().__init__()
        self.output_path = Path(output_path)
        self.overlap = overlap
        self.delta = delta
        self.blend_method = blend_method
        self.chunk_size = chunk_size
        self.cog_config = cog_config or {}

        self.temp_dir: Path | None = None
        self.patch_metadata: list[dict[str, Any]] = []
        self.num_classes: int | None = None
        self.crs: Any = None
        self.dataset_bounds: tuple[float, float, float, float] | None = None
        self.dataset_res: float | None = None

    def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
        """Initialize state at start of prediction.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        datamodule = trainer.datamodule
        if hasattr(datamodule, 'predict_dataset'):
            dataset = datamodule.predict_dataset
            if hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            self.crs = getattr(dataset, 'crs', None)
            if hasattr(dataset, 'index') and hasattr(dataset.index, 'bounds'):
                df = dataset.index.bounds
                self.dataset_bounds = (
                    float(df['minx'].min()),
                    float(df['miny'].min()),
                    float(df['maxx'].max()),
                    float(df['maxy'].max()),
                )
            if hasattr(dataset, 'res'):
                self.dataset_res = dataset.res

        self.temp_dir = self.output_path.parent / f'.tmp_{self.output_path.stem}'
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.patch_metadata = []

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Save patch predictions to disk.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
            outputs: Outputs from predict_step.
            batch: Current batch.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
        """
        logits = outputs['logits']
        bounds = outputs.get('bounds')
        transforms = outputs.get('transform')

        if bounds is None:
            raise ValueError(
                'batch["bounds"] is None. Ensure you are using PR #3138 changes.'
            )
        if transforms is None:
            raise ValueError(
                'batch["transform"] is None. Ensure you are using PR #3140 changes.'
            )

        if self.num_classes is None:
            self.num_classes = logits.shape[1]

        batch_size = logits.shape[0]
        for i in range(batch_size):
            patch_id = batch_idx * batch_size + i
            patch_logits = logits[i].cpu().clone()
            bounds_tensor = bounds[i].cpu().clone()
            transform_tensor = transforms[i].cpu().clone()

            assert self.temp_dir is not None
            patch_path = self.temp_dir / f'patch_{patch_id:06d}.pt'
            torch.save(
                {
                    'logits': patch_logits,
                    'bounds': bounds_tensor,
                    'transform': transform_tensor,
                },
                patch_path,
            )

            geo_xmin = bounds_tensor[0].item()
            geo_xmax = bounds_tensor[1].item()
            geo_ymin = bounds_tensor[3].item()
            geo_ymax = bounds_tensor[4].item()

            self.patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_path,
                    'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                    'transform': transform_tensor,
                }
            )

    def on_predict_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Merge patches and write GeoTIFF.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        from torchgeo.inference.blending import weighted_merge

        if not self.patch_metadata:
            raise ValueError('No patches to merge')

        assert self.num_classes is not None
        weighted_merge(
            patch_metadata=self.patch_metadata,
            num_classes=self.num_classes,
            overlap=self.overlap,
            delta=self.delta,
            blend_method=self.blend_method,
            crs=self.crs,
            output_path=self.output_path,
            chunk_size=self.chunk_size,
            cog_config=self.cog_config,
            dataset_bounds=self.dataset_bounds,
            dataset_res=self.dataset_res,
        )

        if self.temp_dir is not None and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
