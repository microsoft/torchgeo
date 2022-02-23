# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars dataset."""

import glob
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import VisionDataset
from .utils import download_url, extract_archive


class USAVars(VisionDataset):
    """USAVars dataset.

    Dataset format:
    * images are 4-channel tifs
    * labels are singular float values

    Dataset labels:
    - tree cover
    - elevation
    - population density
    - nighttime lights
    - income per houshold
    - road length
    - housing price

    .. versionadded:: 0.3
    """

    csv_prefix = (
        "https://files.codeocean.com/files/verified/"
        + "fa908bbc-11f9-4421-8bd3-72a4bf00427f_v2.0/data/int/applications/"
    )
    csv_postfix = "_CONTUS_16_640_POP_100000_0.csv?download"

    data_url = "https://mosaiks.blob.core.windows.net/datasets/uar.zip"
    dirname = "usavars"
    zipfile = dirname + ".zip"

    md5 = "677e89fd20e5dd0fe4d29b61827c2456"

    label_urls = {
        "housing": csv_prefix + "housing/outcomes_sampled_housing" + csv_postfix,
        "income": csv_prefix + "income/outcomes_sampled_income" + csv_postfix,
        "roads": csv_prefix + "roads/outcomes_sampled_roads" + csv_postfix,
        "nightligths": csv_prefix
        + "nightlights/outcomes_sampled_nightlights"
        + csv_postfix,
        "population": csv_prefix
        + "population/outcomes_sampled_population"
        + csv_postfix,
        "elevation": csv_prefix + "elevation/outcomes_sampled_elevation" + csv_postfix,
        "treecover": csv_prefix + "treecover/outcomes_sampled_treecover" + csv_postfix,
    }

    def __init__(
        self, root: str = "data", download: bool = False, checksum: bool = False
    ) -> None:
        """Initialize a new USAVars dataset instance."""
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, float]]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample = self.files[index]
        sample["image"] = self._load_image(sample["image"])
        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.root, self.dirname, "uar")
        files = os.listdir(file_path)

        files = files[
            :10
        ]  # TODO: remove this, keeping temporarily because this func is very slow

        # csvs = self.label_urls.keys() # only uar for now
        csvs = ["treecover", "elevation", "population"]
        labels_ds = [
            (lab, pd.read_csv(os.path.join(self.root, lab + ".csv"))) for lab in csvs
        ]
        samples = []
        for f in files:
            img_path = os.path.join(file_path, f)
            samp = {"image": img_path}

            id_ = f[5:-4]

            for lab, ds in labels_ds:
                samp[lab] = ds[ds["ID"] == id_][lab].values[0]

            samples.append(samp)
        return samples

    def _load_image(self, path: str) -> Tensor:
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read()
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.dirname)
        if glob.glob(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile)
        if glob.glob(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        self._download()
        self._extract()

    def _download(self) -> None:
        for f_name in self.label_urls:
            download_url(self.label_urls[f_name], self.root, filename=f_name + ".csv")
        download_url(
            self.data_url,
            self.root,
            filename=self.zipfile,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        src = os.path.join(self.root, self.zipfile)
        dst = os.path.join(self.root, self.dirname)
        extract_archive(src, dst)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_labels: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_labels: flag indicating whether to show labels above panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"][:3].numpy()  # get RGB inds
        image = np.moveaxis(image, 0, 2)

        fig, axs = plt.subplots(figsize=(10, 10))
        axs.imshow(image)
        axs.axis("off")

        if show_labels:
            labels = [(lab, val) for lab, val in sample.items() if lab != "image"]
            label_string = ""
            for lab, val in labels:
                label_string += f"{lab}={val} "

            axs.set_title(label_string)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
