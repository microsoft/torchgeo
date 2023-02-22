"""MapInWild dataset."""

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

from torch import Tensor
from typing import Callable, Optional, Dict, Tuple
import rasterio
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from .geo import NonGeoDataset
from .utils import check_integrity, extract_archive, download_url

class MapInWild(NonGeoDataset):
    """MapInWild dataset.

        The `MapInWild <https://arxiv.org/abs/2212.02265>`_ is curated for the task of wilderness mapping on a pixel-level. 
        MapInWild is a multi-modal dataset and comprises various geodata acquired and formed from different RS sensors over 1018 locations:
        dual-pol Sentinel-1, four-season Sentinel-2 with 10 bands, ESA WorldCover map, and Visible Infrared Imaging Radiometer Suite NightTime Day/Night band.
        The dataset consists of 8144 images with the shape of 1920 × 1920 pixels. 
        The images are weakly annotated from the World Database of Protected Areas (WDPA).

        Dataset features:
        * 1018 areas globally sampled from the WDPA 
        * 10-Band Sentinel-2
        * Dual-pol Sentinel-1 
        * ESA WorldCover Land Cover 
        * Visible Infrared Imaging Radiometer Suite NightTime Day/Night Band

        If you use this dataset in your research, please cite the following paper:

        * https://arxiv.org/abs/2212.02265
    """
    BAND_SETS: Dict[str, Tuple[str, ...]] = {
        "all": (
            "VV",
            "VH",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
            "2020_Map",
            "avg_rad"
            ), 
        "s1": ("VV", "VH"),
        "s2": (
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12"
        ),
        "esa_wc": ("2020_Map",),
        "viirs": ("avg_rad",),
    }

    band_names = (
        "VV",
        "VH",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "2020_Map",
        "avg_rad"
    )

    rgb_bands = ["B4", "B3", "B2"]
    modality_urls = {
    "esa_wc": {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/esa_wc/ESA_WC.zip"},
    "viirs": {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/viirs/VIIRS.zip"},
    "mask": {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/mask/mask.zip"},
    "s1" : {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s1/s1_part1.zip",
            "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s1/s1_part2.zip"},
    "s2_temporal_subset" : {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_temporal_subset/s2_temporal_subset_part1.zip",
                            "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_temporal_subset/s2_temporal_subset_part2.zip"},
    "s2_autumn" : {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_autumn/s2_autumn_part1.zip",
                   "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_autumn/s2_autumn_part2.zip"},
    "s2_spring" :  {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_spring/s2_spring_part1.zip",
                   "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_spring/s2_spring_part2.zip"},
    "s2_summer" : {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_summer/s2_summer_part1.zip",
                    "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_summer/s2_summer_part2.zip"},
     "s2_winter" : {"https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_winter/s2_winter_part1.zip",
                    "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/s2_winter/s2_winter_part2.zip"}
    }
    
    split_url = "https://huggingface.co/datasets/burakekim/mapinwild/resolve/main/split_IDs/split_IDs.csv"
    main_directory = os.path.join(os.getcwd(), "data")
    mask_palette = {1 : (0,153,0),
                0 : (255,255,255)} 

    def __init__(
        self,
        root: str = "data",
        modality: list = ["mask","esa_wc","s2_temporal_subset","viirs","s1"],
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False
    ) -> None:
        """Initialize a new MapInWild dataset instance.

        Args:
            root: root directory where dataset can be found
            modality: the modality to download
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            
        Raises:
            AssertionError: if ``split`` argument is invalid
        """
        assert split in ["train", "validation", "test"]

        self.checksum = checksum
        self.root = root
        self.transforms = transforms
        self.modality = modality
        self.modality.remove("mask")

        self.download = download
        self._verify_split() 

        split_dataframe = pd.read_csv(os.path.join(self.root, "split_IDs.csv"))
        self.ids = split_dataframe[split].dropna().values.tolist() 
        self.ids = [int(i) for i in self.ids]
        
        #Check if the requested list of modalities exist in the directory  
        if not set(self.modality).issubset(set(os.listdir(self.main_directory))):
            for modal in modality: 
                for modality_link in self.modality_urls[modal]:
                    self._verify(modality_link)
                
            for modal in modality:
                if len(self.modality_urls[modal]) > 1:
                    self.merge_parts(self.main_directory, modal)

        self.list_modals = []
        
    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        id = self.ids[index]
        
        mask = self._load_raster(id, "mask")
        mask[mask != 0] = 1 

        for modals in self.modality: 
            modal = self._load_raster(id, modals)

            if self.transforms is not None:
                modal = self.transforms(modal)
                mask = self.transforms(mask)

            self.list_modals.append(modal)

                        
        image = torch.cat(self.list_modals, dim=0)
        return image, mask

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_raster(self, filename: str, source: str) -> Tensor:
        """Load a single raster image or target.

        Args:
            filename: name of the file to load
            source: the filename of the modality

        Returns:
            the raster image or target
        """
        with rasterio.open(
                os.path.join(self.root,
                                "{}".format(source), 
                                "{}.tif".format(filename),
                )
        ) as f:
            array = f.read()
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            tensor = torch.from_numpy(array)
            return tensor
            
    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename), self.md5 if self.checksum else None
        )
        return integrity

    def _verify(self, url) -> None:
        """Verify the integrity of the dataset."""
        # Download and extract the dataset
        self._download(url)
        self._extract(url)

    def _verify_split(self) -> None:
        """Verify the integrity of the split file."""
        # Download the dataset
        download_url(
            self.split_url,
            self.root,
            filename=os.path.split(self.split_url)[1],
            md5=self.md5 if self.checksum else None,
        )

    def _download(self, url) -> None:
        """Download the dataset."""
        download_url(
            url,
            self.root,
            filename=os.path.split(url)[1],
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self, url) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, os.path.split(url)[1])
        extract_archive(filepath)

    @staticmethod
    def merge_parts(source_path, modality):
        """Merge the modalities that are downloaded and extracted in parts."""
        fname_p1 = modality + "_part1"
        fname_p2 = modality + "_part2"
        source_folder = os.path.join(source_path, fname_p1) 
        destination_folder = os.path.join(source_path, fname_p2) 

        for file_name in os.listdir(source_folder):
            source = os.path.join(source_folder, file_name) 
            destination = os.path.join(destination_folder, file_name)
            if os.path.isfile(source):
                shutil.move(source, destination)

        shutil.rmtree(source_folder)
        dest_split = os.path.split(destination_folder)
        if modality == 's2_temporal_subset':
            rename_dest = os.path.join(dest_split[0], 's2_temporal_subset')
        else:
            rename_dest = os.path.join(dest_split[0], dest_split[1].split('_')[0])
        os.rename(destination_folder, rename_dest)
        
    @staticmethod
    def convert_to_binary(arr_2d, palette):
        """ Numeric labels to RGB-color encoding."""
        arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

        for c, i in palette.items():
            m = arr_2d == c
            arr_3d[m] = i
        return arr_3d

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
            
        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        mask = sample["mask"].numpy().squeeze()
        color_mask = self.convert_to_binary(mask,self.mask_palette)
        num_panels = 2
        showing_predictions = "prediction" in sample

        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(color_mask, interpolation="none")
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(
                predictions, vmin=0, vmax=1, interpolation="none"
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig