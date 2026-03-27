# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tiled inference callback for semantic segmentation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

import rasterio
import torch
from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback
from rasterio.transform import Affine


class TiledInferenceCallback(Callback):
    r"""Callback for tiled inference with weighted blending.

    Enables large-area inference by splitting a scene into overlapping patches,
    running predictions on each, and stitching results into a single
    Cloud-Optimized GeoTIFF.

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────────────────────┐
        │                         PREDICTION LIFECYCLE                            │
        ├─────────────────────────────────────────────────────────────────────────┤
        │                                                                         │
        │  ┌───────────────────────────────────────────────────────────────────┐  │
        │  │  Phase 1: on_predict_start()                                      │  │
        │  │                                                                   │  │
        │  │  - Disables Lightning's prediction storage (memory leak fix)      │  │
        │  │  - Introspects datamodule.predict_dataset for CRS, bounds, res    │  │
        │  │  - Creates temp directory for patch files                         │  │
        │  └───────────────────────────────────────────────────────────────────┘  │
        │                              │                                          │
        │                              ▼                                          │
        │  ┌───────────────────────────────────────────────────────────────────┐  │
        │  │  Phase 2: on_predict_batch_end()    (called per batch)            │  │
        │  │                                                                   │  │
        │  │  For each sample in the batch:                                    │  │
        │  │    1. Extract probabilities, bounds, transform from predict_step  │  │
        │  │    2. Convert softmax → argmax → one-hot (float32 → uint8)        │  │
        │  │    3. Write as georeferenced GeoTIFF to temp dir                  │  │
        │  │    4. Collect metadata: patch_id, file path, geo_bbox, transform  │  │
        │  │                                                                   │  │
        │  │  Temp patch layout:                                               │  │
        │  │    .tmp_predictions/                                              │  │
        │  │      patch_000000.tif   <- one-hot uint8, LZW compressed          │  │
        │  │      patch_000001.tif      with affine transform + CRS            │  │
        │  │      ...                                                          │  │
        │  └───────────────────────────────────────────────────────────────────┘  │
        │                              │                                          │
        │                              ▼                                          │
        │  ┌───────────────────────────────────────────────────────────────────┐  │
        │  │  Phase 3: on_predict_epoch_end()                                  │  │
        │  │                                                                   │  │
        │  │  weighted_merge():                                                │  │
        │  │    1. Reconstruct scene geometry                                  │  │
        │  │       - Use dataset_bounds/res if available                       │  │
        │  │       - Otherwise infer from patch transforms                     │  │
        │  │                                                                   │  │
        │  │    2. Build grid spatial index over all patches                   │  │
        │  │       ┌─────┬─────┬─────┐                                         │  │
        │  │       │ p0  │ p1  │ p2  │  Grid cells map to patch lists          │  │
        │  │       ├─────┼─────┼─────┤  for O(1) lookup per chunk              │  │
        │  │       │ p3  │ p4  │ p5  │                                         │  │
        │  │       └─────┴─────┴─────┘                                         │  │
        │  │                                                                   │  │
        │  │    3. Process output in chunks (default 4096 x 4096 px):          │  │
        │  │       For each chunk:                                             │  │
        │  │         a. Query spatial index → overlapping patches              │  │
        │  │         b. Read each patch from disk                              │  │
        │  │         c. Apply edge-aware delta cropping:                       │  │
        │  │            - Interior edges: crop delta px (remove artifacts)     │  │
        │  │            - Scene boundary edges: crop 0 (preserve coverage)     │  │
        │  │         d. Generate blend mask (cosine or linear ramp):           │  │
        │  │                                                                   │  │
        │  │            1.0 ┤  ╭────────────────╮                              │  │
        │  │                │ /                  \   <- cosine ramp in overlap  │  │
        │  │            0.0 ┤/                    \                            │  │
        │  │                └──┬──────────────┬──┘                             │  │
        │  │                overlap        overlap                             │  │
        │  │                                                                   │  │
        │  │         e. Accumulate: output += patch x mask; weights += mask    │  │
        │  │         f. Normalize: output /= weights                           │  │
        │  │         g. Argmax → uint8 class labels                            │  │
        │  │         h. Write chunk to output via GeoTIFFWriter                │  │
        │  │                                                                   │  │
        │  │    4. GeoTIFFWriter.finalize()                                    │  │
        │  │       - Build COG overviews [2, 4, 8, 16, 32, 64]                 │  │
        │  │                                                                   │  │
        │  │    5. Clean up temp directory                                     │  │
        │  └───────────────────────────────────────────────────────────────────┘  │
        │                                                                         │
        │  Output: Cloud-Optimized GeoTIFF with class labels, CRS, overviews      │
        └─────────────────────────────────────────────────────────────────────────┘

    Example::

        from lightning import Trainer
        from torchgeo.callbacks import TiledInferenceCallback

        callback = TiledInferenceCallback(
            output_path='predictions.tif', overlap=32, delta=8
        )
        trainer = Trainer(callbacks=[callback])
        trainer.predict(task, datamodule)

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        output_path: str | Path,
        overlap: int = 32,
        delta: int = 8,
        blend_method: Literal['cosine', 'linear'] = 'cosine',
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
                Defaults to building overviews at levels [2, 4, 8, 16, 32, 64]
                with nearest resampling and LZW compression.
        """
        super().__init__()
        self.output_path = Path(output_path)
        self.overlap = overlap
        self.delta = delta
        self.blend_method = blend_method
        self.chunk_size = chunk_size

        # Default COG config with overviews (using rasterio)
        default_cog_config: dict[str, Any] = {
            'overviews': [2, 4, 8, 16, 32, 64],
            'overview_resampling': 'nearest',
            'compress': 'lzw',
        }
        self.cog_config = {**default_cog_config, **(cog_config or {})}

        self.temp_dir: Path | None = None
        self.patch_metadata: list[dict[str, Any]] = []
        self.num_classes: int | None = None
        self.crs: Any = None
        self.dataset_bounds: tuple[float, float, float, float] | None = None
        self.dataset_res: float | None = None

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize state at start of prediction.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        if hasattr(trainer, 'predict_loop'):
            trainer.predict_loop.return_predictions = False

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
        trainer: Trainer,
        pl_module: LightningModule,
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
        probabilities = outputs['probabilities']
        bounds = outputs.get('bounds')
        transforms = outputs.get('transform')

        if bounds is None:
            raise ValueError(
                'outputs["bounds"] is None; ensure predict_step returns bounds metadata.'
            )
        if transforms is None:
            raise ValueError(
                'outputs["transform"] is None; ensure predict_step returns transform metadata.'
            )

        if self.num_classes is None:
            self.num_classes = probabilities.shape[1]

        batch_size = probabilities.shape[0]
        for i in range(batch_size):
            patch_id = len(self.patch_metadata)
            patch_probabilities = probabilities[i].cpu().clone()
            bounds_tensor = bounds[i].cpu().clone()
            transform_tensor = transforms[i].cpu().clone()

            assert self.temp_dir is not None
            patch_path = self.temp_dir / f'patch_{patch_id:06d}.tif'
            num_classes = patch_probabilities.shape[0]
            class_predictions = patch_probabilities.argmax(dim=0)
            one_hot = (
                torch.nn.functional.one_hot(
                    class_predictions.long(), num_classes=num_classes
                )
                .permute(2, 0, 1)
                .to(torch.uint8)
                .numpy()
            )
            transform_list = transform_tensor.tolist()
            patch_transform = Affine(*transform_list)
            with rasterio.open(
                patch_path,
                'w',
                driver='GTiff',
                height=one_hot.shape[1],
                width=one_hot.shape[2],
                count=one_hot.shape[0],
                dtype='uint8',
                compress='lzw',
                tiled=True,
                transform=patch_transform,
                crs=self.crs,
            ) as dst:
                dst.write(one_hot)

            geo_xmin = bounds_tensor[0].item()
            geo_xmax = bounds_tensor[1].item()
            geo_ymin = bounds_tensor[3].item()
            geo_ymax = bounds_tensor[4].item()

            self.patch_metadata.append(
                {
                    'patch_id': patch_id,
                    'file': patch_path,
                    'geo_bbox': (geo_xmin, geo_ymin, geo_xmax, geo_ymax),
                    'transform': transform_list,
                }
            )

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Merge patches and write GeoTIFF.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        from torchgeo.callbacks.blending import weighted_merge

        if not self.patch_metadata:
            raise ValueError('No patches to merge')

        assert self.num_classes is not None
        try:
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
        finally:
            if self.temp_dir is not None and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
