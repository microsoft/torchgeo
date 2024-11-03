# Code for loading dataset licensed under the MIT License.
#
# FLAIR dataset is realeasd under open license 2.0
# ..... https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf
# ..... https://ignf.github.io/FLAIR/#FLAIR2
#


"""
FLAIR2 dataset.
"""

import glob
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Optional

import einops
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
from torchgeo.datasets.errors import DatasetNotFoundError, RGBBandsMissingError
from torchgeo.transforms.transforms import _RandomNCrop
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datasets.geo import IntersectionDataset, RasterDataset, NonGeoDataset
from torchgeo.datasets.utils import Path, download_url, extract_archive, BoundingBox
from torchgeo.datamodules import NonGeoDataModule, GeoDataModule
from kornia.augmentation.container import AugmentationSequential
import kornia.augmentation as K
from torch.utils.data import random_split


class FLAIR2(NonGeoDataset):
    splits: ClassVar[Sequence[str]] = ('train', 'test')
    
    url_prefix: ClassVar[str] = 'https://ign-public-data.s3.eu-west-2.amazonaws.com/FLAIR2/'
    # TODO: add checksums for safety
    md5: str = ""
    
    dir_names: dict[dict[str, str]] = {
        "train": {
            "images": "flair_aerial_train",
            "masks": 'flair_labels_train',
        },
        "test": {
            "images": "flair_2_aerial_test",
            "masks": 'flair_2_labels_test',
        }
    }
    globs: dict[str, str] = {
        "images": "IMG_*.tif",
        "masks": "MSK_*.tif",
    }

    # Band information
    rgb_bands: tuple = ("B01", "B02", "B03")
    all_bands: tuple = ("B01", "B02", "B03", "B04", "B05")

    classes: tuple[str] = (
        "building",
        "pervious surface",
        "impervious surface",
        "bare soil",
        "water",
        "coniferous",
        "deciduous",
        "brushwood",
        "vineyard",
        "herbaceous vegetation",
        "agricultural land",
        "plowed land",
        "other"
    )

    statistics = {
        "train":{
            "B01": {
                "min": 0.0,
                "max": 255.0,
                "mean": 113.77526983072,
                "stdv": 1.4678962001526,
            },
            "B02": {
                "min": 0.0,
                "max": 255.0,
                "mean": 118.08112962721,
                "stdv": 1.2889349378677,
            },
            "B03": {
                "min": 0.0,
                "max": 255.0,
                "mean": 109.27393364381,
                "stdv": 1.2674219560871,
            },
            "B04": {
                "min": 0.0,
                "max": 255.0,
                "mean": 102.36417944851,
                "stdv": 1.1057592647291,
            },
            "B05": {
                "min": 0.0,
                "max": 255.0,
                "mean": 16.697295721745,
                "stdv": 0.82764953440507,
            },
        }
    }

    @staticmethod
    def per_band_statistics(split: str, bands: Sequence[str] = all_bands) -> tuple[list[float]]:
        # TODO: filter used bands
        """Get statistics (min, max, means, stdvs) for each band

        Args:
            split (str): Split for which to get statistics (currently only for train)
            bands (Sequence[str], optional): _description_. Defaults to all_bands.

        Returns:
            tuple[list[float]]: _description_
        """
        assert split in FLAIR2.statistics.keys(), f"Statistics for '{split}' not available; use: '{list(FLAIR2.statistics.keys())}'"
        ordered_bands_statistics = FLAIR2.statistics[split]
        ordered_bands_statistics = list(dict(filter(lambda keyval: keyval[0] in bands, ordered_bands_statistics.items())).values())
        mins = list(map(lambda dict: dict["min"], ordered_bands_statistics))
        maxs = list(map(lambda dict: dict["max"], ordered_bands_statistics))
        means = list(map(lambda dict: dict["mean"], ordered_bands_statistics))
        stdvs = list(map(lambda dict: dict["stdv"], ordered_bands_statistics))
        return mins, maxs, means, stdvs
    
    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        # FLAIR multiplies height data by 5 for storage, argument for not doing this
        remove_height_scaling: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new FLAIR2 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: which split to load, one of 'train' or 'test'
            bands: which bands to load (B01, B02, B03, B04, B05)
            transforms: optional transforms to apply to sample
            download: whether to download the dataset if it is not found
            checksum: whether to verify the dataset using checksums
            

        Raises:
            DatasetNotFoundError
        """
        assert split in self.splits, f"Split '{split}' not in supported splits: '{self.splits}'"

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.bands = bands

        self._verify()

        self.files = self._load_files()
    
    def get_num_bands(self) -> int:
        """Return the number of bands in the dataset.

        Returns:
            int: number of bands in the initialized dataset (might vary from all_bands)
        """
        return len(self.bands)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # TODO: add sentinel-2 bands
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and mask at that index with image of dimension get_num_bands()x512x512
            and mask of dimension 512x512
        """
        aerial_fn = self.files[index]["image"]
        mask_fn = self.files[index]["mask"]

        aerial = self._load_image(aerial_fn)
        mask = self._load_target(mask_fn)

        image = aerial 
        sample = {'image': image, 'mask': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.files)

    def _load_files(self) -> list[dict[str, str]]:
        # TODO: add loading of sentinel-2 files
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image, masks
        """
        images = sorted(glob.glob(os.path.join(
            self.root, 
            self.dir_names[self.split]["images"],
            "**", self.globs["images"]), recursive=True))
        masks = sorted(glob.glob(os.path.join(
            self.root, 
            self.dir_names[self.split]["masks"],
            "**", self.globs["masks"]), recursive=True))
        
        files = [
            dict(image=image, mask=mask)
            for image, mask in zip(images, masks)
        ]
        
        return files

    def _load_image(self, path: Path) -> Tensor:
        # TODO: add loading of sentinel-2 images if requested
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the loaded image
        """
        filename = os.path.join(path)
        with rasterio.open(filename) as f:
            array: np.typing.NDArray[np.int_] = f.read()
            tensor = torch.from_numpy(array).float()
            # TODO: handle storage optimized format for height data
            if "B05" in self.bands:
                # Height channel will always be the last dimension
                tensor[-1] = torch.div(tensor[-1], 5)
            
        return tensor

    def _load_target(self, path: Path) -> Tensor:
        """Load a single mask corresponding to image.

        Args:
            path: path to the mask

        Returns:
            the mask of the image
        """
        filename = os.path.join(path)
        with rasterio.open(filename) as f:
            array: np.typing.NDArray[np.int_] = f.read(1)
            tensor = torch.from_numpy(array).long()
            # TODO: check if rescaling is smart (i.e. datapaper explains differently -> confusion?)
            # According to datapaper, the dataset contains classes beyond 13
            # however, those are grouped into a single "other" class
            # Rescale the classes to be in the range [0, 12] by subtracting 1
            torch.clamp(tensor - 1, 0, len(self.classes) - 1, out=tensor)
            #torch.clamp(tensor, 0, len(self.classes), out=tensor)
            
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        
        # Files to be extracted
        to_extract: list = []

        # Check if dataset files (by checking glob) are present already
        for train_or_test, dir_name in self.dir_names[self.split].items(): 
            downloaded_path = os.path.join(self.root, dir_name)
            if not os.path.isdir(downloaded_path):
                to_extract.append(dir_name)
                continue

            files_glob = os.path.join(downloaded_path, "**", self.globs[train_or_test])
            if not glob.glob(files_glob, recursive=True):
                to_extract.append(dir_name)
        
        if not to_extract:
            print("Data has been downloaded and extracted already...") 
            return

        # Deepcopy files to be extracted and check wether the zip is downloaded
        to_download = list(map(lambda x: x, to_extract))
        for candidate in to_extract:
            zipfile = os.path.join(self.root, f"{candidate}.zip")
            if glob.glob(zipfile):
                print(f"Extracting: {candidate}")
                self._extract(candidate)
                to_download.remove(candidate)
        
        # Check if there are still files to download
        if not to_download: 
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        print("Downloading: ", to_download)
        for candidate in to_download:
            self._download(candidate)
            self._extract(candidate)

    def _download(self, url: str) -> None:
        """Download the dataset."""
        download_url(
            os.path.join(self.url_prefix, f"{url}.zip"), self.root
        )

    def _extract(self, file_path: str) -> None:
        """Extract the dataset."""
        assert isinstance(self.root, str | os.PathLike)
        zipfile = os.path.join(self.root, f"{file_path}.zip")
        extract_archive(zipfile)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        # TODO: plot sentinel image too
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample['mask'].numpy().astype('uint8').squeeze()

        num_panels = 2
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy().astype('uint8').squeeze()
            num_panels += 1

        kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 4, 'interpolation': 'none'}
        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask, **kwargs)
        axs[1].axis('off')
        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_predictions:
            axs[2].imshow(predictions, **kwargs)
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig