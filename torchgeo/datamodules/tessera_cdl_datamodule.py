# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tessera + CDL TorchGeo DataModule for pixel-wise classification."""

from pathlib import Path

from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import CDL, GeoDataset, TesseraEmbeddings
from torchgeo.datasets.utils import Sample
from torchgeo.samplers import GriddedPatchSampler, RandomPatchSampler

from .utils import collate_fn_embeddings


class TesseraCDLDataModule(GeoDataModule):
    """DataModule for Tessera + CDL dataset.

    .. versionadded:: 0.10
    """

    train_dataset: GeoDataset | None
    val_dataset: GeoDataset | None
    test_dataset: GeoDataset | None

    def __init__(
        self,
        tessera_root: str,
        data_dir: str = './data',
        year: int = 2024,
        classes: list[int] | None = None,
        patch_size: int = 32,
        num_train_patches: int = 1000,
        batch_size: int = 4,
        num_workers: int = 0,
        download: bool = True,
        subfolder: str = 'global_0.1_degree_representation',
    ) -> None:
        """Initialize a new TesseraCDLDataModule instance.

        Args:
            tessera_root: Root directory containing Tessera embeddings, laid
                out as ``<tessera_root>/<split>/<subfolder>/<year>``.
            data_dir: Root directory containing CDL data.
            year: Year of CDL data to use.
            classes: List of class indices to include. If None, all classes are included.
            patch_size: Size of patches to extract.
            num_train_patches: Number of training patches to sample.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            download: Whether to download data if not found locally.
            subfolder: Subdirectory inside `tessera_root` containing Tessera
                embeddings (default: 'global_0.1_degree_representation').
        """
        super().__init__(dataset_class=GeoDataset)

        self.tessera_root = tessera_root
        self.data_dir = data_dir
        self.year = year
        self.classes = classes
        self.patch_size = patch_size
        self.num_train_patches = num_train_patches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.subfolder = subfolder
        self.collate_fn = collate_fn_embeddings

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Return the batch unaltered.

        ``collate_fn_embeddings`` flattens each batch into per-pixel
        ``embeddings``/``labels`` pairs, so the image-based Kornia
        augmentations applied by the base class do not apply here.

        Args:
            batch: A batch of data.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            The batch, unmodified.
        """
        return batch

    def _build_dataset(self, split: str, cdl: CDL) -> GeoDataset:
        """Build a dataset for the given split.

        Args:
            split: One of 'train', 'val', or 'test'.
            cdl: The shared CDL dataset to intersect with.

        Returns:
            The intersection of the Tessera embeddings and CDL datasets.

        Raises:
            FileNotFoundError: If the split's Tessera directory does not exist.
        """
        split_dir = Path(self.tessera_root) / split / self.subfolder / str(self.year)

        if not split_dir.exists():
            raise FileNotFoundError(f'Tessera directory not found: {split_dir}')

        tessera = TesseraEmbeddings(paths=str(split_dir))
        return tessera & cdl

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets and samplers."""
        cdl_classes = (
            self.classes if self.classes is not None else list(CDL.valid_classes)
        )

        cdl = CDL(
            paths=self.data_dir,
            years=[self.year],
            classes=cdl_classes,
            download=self.download,
        )

        self.train_dataset = self._build_dataset('train', cdl)

        if stage in ['fit']:
            self.train_sampler = RandomPatchSampler(
                self.train_dataset, size=self.patch_size, length=self.num_train_patches
            )

        if stage in ['fit', 'validate']:
            self.val_dataset = self._build_dataset('val', cdl)
            self.val_sampler = GriddedPatchSampler(
                self.val_dataset, size=self.patch_size, stride=self.patch_size
            )

        if stage in ['test']:
            self.test_dataset = self._build_dataset('test', cdl)
            self.test_sampler = GriddedPatchSampler(
                self.test_dataset, size=self.patch_size, stride=self.patch_size
            )
