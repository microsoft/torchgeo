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
        train_dir: str | None = None,
        val_dir: str | None = None,
        test_dir: str | None = None,
    ) -> None:
        """Initialize a new TesseraCDLDataModule instance.

        Args:
            tessera_root: Root directory containing Tessera embeddings.
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
            train_dir: Optional explicit directory containing the training
                Tessera embeddings. If not provided, the default
                ``<tessera_root>/train/<subfolder>/<year>`` layout is used.
            val_dir: Optional explicit directory containing the validation
                Tessera embeddings. If not provided, the default
                ``<tessera_root>/val/<subfolder>/<year>`` layout is used.
            test_dir: Optional explicit directory containing the test Tessera
                embeddings. If not provided, the default
                ``<tessera_root>/test/<subfolder>/<year>`` layout is used.
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
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
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

    def _build_dataset(self, split: str, cdl: CDL) -> GeoDataset | None:
        """Build a dataset for the given split if the data directory exists."""
        split_dir = None

        if split == 'train' and self.train_dir is not None:
            split_dir = Path(self.train_dir)
        elif split == 'val' and self.val_dir is not None:
            split_dir = Path(self.val_dir)
        elif split == 'test' and self.test_dir is not None:
            split_dir = Path(self.test_dir)
        else:
            base_path = Path(self.tessera_root)
            split_dir = base_path / split / self.subfolder / str(self.year)

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

        train_dataset = self._build_dataset('train', cdl)
        if train_dataset is None:
            raise FileNotFoundError('Train dataset could not be built')
        self.train_dataset = train_dataset

        if stage in ['fit']:
            self.train_sampler = RandomPatchSampler(
                self.train_dataset, size=self.patch_size, length=self.num_train_patches
            )

        if stage in ['fit', 'validate']:
            val_dataset = self._build_dataset('val', cdl)
            self.val_dataset = val_dataset
            if self.val_dataset is not None:
                self.val_sampler = GriddedPatchSampler(
                    self.val_dataset, size=self.patch_size, stride=self.patch_size
                )

        if stage in ['test']:
            test_dataset = self._build_dataset('test', cdl)
            self.test_dataset = test_dataset
            if self.test_dataset is not None:
                self.test_sampler = GriddedPatchSampler(
                    self.test_dataset, size=self.patch_size, stride=self.patch_size
                )
