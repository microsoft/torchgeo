# Copyright (c) TorchGeo Contributors
# All rights reserved.
# Licensed under the MIT License.

"""Tiled inference callback for semantic segmentation."""

from __future__ import annotations

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
        blend_method: str = 'cosine',
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

        self.temp_dir = self.output_path.parent / f'.tmp_{self.output_path.stem}'
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.patch_metadata = []

        print(f'Tiled inference: saving patches to {self.temp_dir}')
        if self.crs:
            print(f'Tiled inference: using CRS {self.crs}')

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
            patch_logits = logits[i].cpu()
            bounds_tensor = bounds[i].cpu()
            transform_tensor = transforms[i].cpu()

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

            x_start = int(bounds_tensor[0].item())
            x_stop = int(bounds_tensor[1].item())
            y_start = int(bounds_tensor[3].item())
            y_stop = int(bounds_tensor[4].item())

            self.patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_path,
                    'bbox': (x_start, y_start, x_stop, y_stop),
                    'transform': transform_tensor,
                }
            )

        if batch_idx % 10 == 0:
            print(f'Processed {len(self.patch_metadata)} patches...')

    def on_predict_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Merge patches and write GeoTIFF.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        from torchgeo.inference.blending import weighted_merge

        if not self.patch_metadata:
            raise ValueError('No patches to merge')

        print(f'Merging {len(self.patch_metadata)} patches...')

        try:
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
            )

            print(f'✅ Inference complete: {self.output_path}')

        finally:
            if self.temp_dir and self.temp_dir.exists():
                import shutil

                shutil.rmtree(self.temp_dir)
