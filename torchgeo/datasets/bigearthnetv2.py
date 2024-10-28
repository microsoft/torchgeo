# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNetv2 dataset."""

import glob
import json
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from rasterio.enums import Resampling
from torch import Tensor
import pandas as pd
from torchvision import transforms
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.utils import download_url, extract_archive, sort_sentinel2_bands

class BigEarthNetv2(NonGeoDataset):
    """BigEarthNet dataset.

    The `BigEarthNet <https://bigearth.net/>`__
    dataset is a dataset for multilabel remote sensing image scene classification.

    Dataset features:

    * 590,326 patches from 125 Sentinel-1 and Sentinel-2 tiles
    * Imagery from tiles in Europe between Jun 2017 - May 2018
    * 12 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 2 synthetic aperture radar bands (120x120 px)
    * 43 or 19 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m

    Dataset classes (43):

    0. Continuous urban fabric
    1. Discontinuous urban fabric
    2. Industrial or commercial units
    3. Road and rail networks and associated land
    4. Port areas
    5. Airports
    6. Mineral extraction sites
    7. Dump sites
    8. Construction sites
    9. Green urban areas
    10. Sport and leisure facilities
    11. Non-irrigated arable land
    12. Permanently irrigated land
    13. Rice fields
    14. Vineyards
    15. Fruit trees and berry plantations
    16. Olive groves
    17. Pastures
    18. Annual crops associated with permanent crops
    19. Complex cultivation patterns
    20. Land principally occupied by agriculture, with significant
        areas of natural vegetation
    21. Agro-forestry areas
    22. Broad-leaved forest
    23. Coniferous forest
    24. Mixed forest
    25. Natural grassland
    26. Moors and heathland
    27. Sclerophyllous vegetation
    28. Transitional woodland/shrub
    29. Beaches, dunes, sands
    30. Bare rock
    31. Sparsely vegetated areas
    32. Burnt areas
    33. Inland marshes
    34. Peatbogs
    35. Salt marshes
    36. Salines
    37. Intertidal flats
    38. Water courses
    39. Water bodies
    40. Coastal lagoons
    41. Estuaries
    42. Sea and ocean

    Dataset classes (19):

    0. Urban fabric
    1. Industrial or commercial units
    2. Arable land
    3. Permanent crops
    4. Pastures
    5. Complex cultivation patterns
    6. Land principally occupied by agriculture, with significant
       areas of natural vegetation
    7. Agro-forestry areas
    8. Broad-leaved forest
    9. Coniferous forest
    10. Mixed forest
    11. Natural grassland and sparsely vegetated areas
    12. Moors, heathland and sclerophyllous vegetation
    13. Transitional woodland, shrub
    14. Beaches, dunes, sands
    15. Inland wetlands
    16. Coastal wetlands
    17. Inland waters
    18. Marine waters

    The source for the above dataset classes, their respective ordering, and
    43-to-19-class mappings can be found here:

    * https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/blob/master/label_indices.json

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2019.8900532

    """  # noqa: E501

    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }

    label_converter = { #TODO: convert to latest table
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }

    # splits_metadata = {
    #     "train": {
    #         "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/train.csv?inline=false",  # noqa: E501
    #         "filename": "bigearthnet-train.csv",
    #         "md5": "623e501b38ab7b12fe44f0083c00986d",
    #     },
    #     "val": {
    #         "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/val.csv?inline=false",  # noqa: E501
    #         "filename": "bigearthnet-val.csv",
    #         "md5": "22efe8ed9cbd71fa10742ff7df2b7978",
    #     },
    #     "test": {
    #         "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/test.csv?inline=false",  # noqa: E501
    #         "filename": "bigearthnet-test.csv",
    #         "md5": "697fb90677e30571b9ac7699b7e5b432",
    #     },
    # }
    splits_metadata = {
        "train": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
        "val": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
        "test": {
            "url": "https://zenodo.org/records/10891137/files/metadata.parquet",
            "filename": "metadata.parquet",
        },
    }
    metadata_locs = {
        "s1": {
            "url": "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst", 
            "md5": "", #unknown
            "filename": "BigEarthNet-S1.tar.zst",
            "directory": "BigEarthNet-S1",
        },
        "s2": {
            "url": "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst", 
            "md5": "", #unknown
            "filename": "BigEarthNet-S2.tar.zst",
            "directory": "BigEarthNet-S2",
        },
        "maps": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "", #unknown
            "filename": "Reference_Maps.zst",
            "directory": "Reference_Maps",
        },
    }
    image_size = (120, 120)

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits_metadata
        assert bands in ["s1", "s2", "all"]
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx_43 = {c: i for i, c in enumerate(self.class_sets[43])}
        self.class2idx_19 = {c: i for i, c in enumerate(self.class_sets[19])} 
        # self._verify()
        self.folders = self._load_folders()

    def class2idx(self, label:str, level=19):
        assert level == 19 or level == 43, "level must be 19 or 43"
        return self.class2idx_19[label] if level == 19 else self.class2idx_43[label]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: dict[str, Tensor] = {"image": image, "label": label}
        # sample: dict[str, Tensor] = {"image": image, "label": None}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)

    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata_locs["s1"]["directory"]
        dir_s2 = self.metadata_locs["s2"]["directory"]
        dir_maps = self.metadata_locs["maps"]["directory"]

        self.metadata = pd.read_parquet(os.path.join(self.root, filename))

        def construct_folder_path(root, dir, patch_id, remove_last:int=2):
            tile_id = '_'.join(patch_id.split('_')[:-remove_last])
            return os.path.join(root, dir, tile_id, patch_id)

        folders = [
            {
                "s1": construct_folder_path(self.root, dir_s1, row['s1_name'] ,3),
                "s2": construct_folder_path(self.root, dir_s2, row['patch_id'],2),
                "maps": construct_folder_path(self.root, dir_maps, row['patch_id'],2),
            }
            for _, row in self.metadata.iterrows()
        ] 

        return folders

    def _load_map_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        folder_maps = self.folders[index]["maps"]
        paths_maps = glob.glob(os.path.join(folder_maps, "*_reference_map.tif"))
        paths_maps = sorted(paths_maps)
        return paths_maps

    def _load_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """

        if self.bands == "all":
            folder_s1 = self.folders[index]["s1"]
            folder_s2 = self.folders[index]["s2"]
            paths_s1 = glob.glob(os.path.join(folder_s1, "*.tif"))
            paths_s2 = glob.glob(os.path.join(folder_s2, "*.tif"))
            paths_s1 = sorted(paths_s1)
            paths_s2 = sorted(paths_s2, key=sort_sentinel2_bands)
            paths = paths_s1 + paths_s2
        elif self.bands == "s1":
            folder = self.folders[index]["s1"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths)
        else:
            folder = self.folders[index]["s2"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths, key=sort_sentinel2_bands)

        return paths

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        return torch.from_numpy(arrays).float()

    def _load_map(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_map_paths(index)
        map = None
        for path in paths:
            with rasterio.open(path) as dataset:
                map = dataset.read(
                    # indexes=1,
                    # out_shape=self.image_size,
                    out_dtype="int32",
                    # resampling=Resampling.bilinear,
                )

        #TODO: convert output values from 43 to 19 classes if needed
        # if self.num_classes == 19:
        #     map = np.vectorize(self.label_converter.get)(map)
        #     map = np.where(map == None, 0, map)

        return torch.from_numpy(map).float()


    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """

        image_labels = self.metadata.iloc[index]['labels']

        # labels -> indices
        indices = [self.class2idx(label) for label in image_labels]

        # Map 43 to 19 class labels: no need, image labels are in 19 class already
        # if self.num_classes == 19:
        #     indices_optional = [self.label_converter.get(idx) for idx in indices]
        #     indices = [idx for idx in indices_optional if idx is not None]

        image_target = torch.zeros(self.num_classes, dtype=torch.long)
        image_target[indices] = 1

        return image_target

    # def _verify(self) -> None:
    #     """Verify the integrity of the dataset.

    #     Raises:
    #         RuntimeError: if ``download=False`` but dataset is missing or checksum fails
    #     """
    #     keys = ["s1", "s2"] if self.bands == "all" else [self.bands]
    #     urls = [self.metadata[k]["url"] for k in keys]
    #     md5s = [self.metadata[k]["md5"] for k in keys]
    #     filenames = [self.metadata[k]["filename"] for k in keys]
    #     directories = [self.metadata[k]["directory"] for k in keys]
    #     urls.extend([self.splits_metadata[k]["url"] for k in self.splits_metadata])
    #     md5s.extend([self.splits_metadata[k]["md5"] for k in self.splits_metadata])
    #     filenames_splits = [
    #         self.splits_metadata[k]["filename"] for k in self.splits_metadata
    #     ]
    #     filenames.extend(filenames_splits)

    #     # Check if the split file already exist
    #     exists = []
    #     for filename in filenames_splits:
    #         exists.append(os.path.exists(os.path.join(self.root, filename)))

    #     # Check if the files already exist
    #     for directory in directories:
    #         exists.append(os.path.exists(os.path.join(self.root, directory)))

    #     if all(exists):
    #         return

    #     # Check if zip file already exists (if so then extract)
    #     exists = []
    #     for filename in filenames:
    #         filepath = os.path.join(self.root, filename)
    #         if os.path.exists(filepath):
    #             exists.append(True)
    #             self._extract(filepath)
    #         else:
    #             exists.append(False)

    #     if all(exists):
    #         return

    #     # Check if the user requested to download the dataset
    #     if not self.download:
    #         raise RuntimeError(
    #             "Dataset not found in `root` directory and `download=False`, "
    #             "either specify a different `root` directory or use `download=True` "
    #             "to automatically download the dataset."
    #         )

    #     # Download and extract the dataset
    #     for url, filename, md5 in zip(urls, filenames, md5s):
    #         self._download(url, filename, md5)
    #         filepath = os.path.join(self.root, filename)
    #         self._extract(filepath)

    def _download(self, url: str, filename: str, md5: str) -> None:
        """Download the dataset.

        Args:
            url: url to download file
            filename: output filename to write downloaded file
            md5: md5 of downloaded file
        """
        if not os.path.exists(filename):
            download_url(
                url, self.root, filename=filename, md5=md5 if self.checksum else None
            )

    def _extract(self, filepath: str) -> None:
        """Extract the dataset.

        Args:
            filepath: path to file to be extracted
        """
        if not filepath.endswith(".csv"):
            extract_archive(filepath)

    def _onehot_labels_to_names(
        self, label_mask: "np.typing.NDArray[np.bool_]"
    ) -> list[str]:
        """Gets a list of class names given a label mask.

        Args:
            label_mask: a boolean mask corresponding to a set of labels or predictions

        Returns
            a list of class names corresponding to the input mask
        """
        labels = []
        for i, mask in enumerate(label_mask):
            if mask:
                labels.append(self.class_sets[self.num_classes][i])
        return labels

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        if self.bands == "s2":
            fig, ax = plt.subplots(1, 2, figsize=(14, 7))
            image = np.rollaxis(sample["imgs"][[3, 2, 1]].numpy(), 0, 3)
            image = np.clip(image / 2000, 0, 1)
            ax[0].imshow(image)
            ax[1].imshow(sample['map'][0],cmap='viridis')

            
        elif self.bands == "all":
            fig, ax = plt.subplots(1, 3, figsize=(14, 7))
            s2 = np.rollaxis(sample["imgs"][[5, 4, 3]].numpy(), 0, 3)
            s2 = np.clip(s2 / 2000, 0, 1)
            ax[0].imshow(s2)
            ax[0].set_axis_off()
            s1 = sample["imgs"][[0, 1]].numpy()
            #create a third channel with difference of VV and VH
            s1 = np.stack((s1[0], s1[1], s1[0] - s1[1]), axis=0)
            #normalize each s1 band
            s1 = np.clip(s1 / -40, 0, 1)
            s1 = np.rollaxis(s1, 0, 3)
            im1 = ax[1].imshow(s1)
            ax[1].set_axis_off()
            im1.set_clim(-20,0)
            ax[2].imshow(sample['map'][0],cmap='viridis')


        elif self.bands == "s1":
            image = sample["image"][0].numpy()

        label_mask = sample["label"].numpy().astype(np.bool_)
        labels = self._onehot_labels_to_names(label_mask)

        showing_predictions = "predictions" in sample
        if showing_predictions:
            prediction_mask = sample["predictions"].numpy().astype(np.bool_)
            predictions = self._onehot_labels_to_names(prediction_mask)

        plt.axis("off")
        if show_titles:
            title = f"Labels: {', '.join(labels)}"
            if showing_predictions:
                title += f"\nPredictions: {', '.join(predictions)}"
            plt.title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
