# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import pandas as pd

from typing import Any, Dict, List

from .utils import (
    download_url,
    extract_archive,
)

class USAVars:
    csv_prefix = "https://files.codeocean.com/files/verified/fa908bbc-11f9-4421-8bd3-72a4bf00427f_v2.0/data/int/applications/"

    data_url = "https://mosaiks.blob.core.windows.net/datasets/uar.zip"
    dirname = "usavars"
    zipfile = dirname + ".zip"

    md5 = "677e89fd20e5dd0fe4d29b61827c2456"

    label_urls = {
        "housing": csv_prefix + "housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv?download",
        "income": csv_prefix + "income/outcomes_sampled_income_CONTUS_16_640_POP_100000_0.csv?download",
        "roads": csv_prefix + "roads/outcomes_sampled_roads_CONTUS_16_640_POP_100000_0.csv?download",
        "nightligths": csv_prefix + "nightlights/outcomes_sampled_nightlights_CONTUS_16_640_POP_100000_0.csv?download",
        "population": csv_prefix + "population/outcomes_sampled_population_CONTUS_16_640_UAR_100000_0.csv?download",
        "elevation": csv_prefix + "elevation/outcomes_sampled_elevation_CONTUS_16_640_UAR_100000_0.csv?download",
        "treecover": csv_prefix + "treecover/outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv?download",
    }

    def __init__(
        self,
        root: str = "data",
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new USAVars dataset instance.
        """

        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> List[Dict[str, Any]]:
        file_path = os.path.join(self.root, self.dirname, "uar")
        files = os.listdir(file_path)

        files = files[:10] # TODO: remove this, keeping temporarily because this func is very slow

        # csvs = self.label_urls.keys() # only uar for now
        csvs = ["treecover", "elevation", "population"]
        labels_ds = [(lab, pd.read_csv(os.path.join(self.root, lab + ".csv"))) for lab in csvs]
        samples = []
        for f in files:
            img_path = os.path.join(file_path, f)
            samp = {"image": img_path}

            id_ = f[5:-4]

            for lab, ds in labels_ds:
                samp[lab] = ds[ds["ID"] == id_][lab].values[0]

            samples.append(samp)
        return samples

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
            download_url(
                self.label_urls[f_name],
                self.root,
                filename=f_name + ".csv",
            )
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
