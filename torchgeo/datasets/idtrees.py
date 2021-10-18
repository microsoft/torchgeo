# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""IDTReeS dataset."""

import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import fiona
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor

from .geo import VisionDataset
from .utils import download_url, extract_archive


class IDTReeS(VisionDataset):
    """IDTReeS dataset.

    The `IDTReeS <https://idtrees.org/competition/>`_
    dataset is a dataset for tree crown detection.

    Dataset classes:

    0. ACPE
    1. ACRU
    2. ACSA3
    3. AMLA
    4. BETUL
    5. CAGL8
    6. CATO6
    7. FAGR
    8. GOLA
    9. LITU
    10. LYLU3
    11. MAGNO
    12. NYBI
    13. NYSY
    14. OXYDE
    15. PEPA37
    16. PIEL
    17. PIPA2
    18. PINUS
    19. PITA
    20. PRSE2
    21. QUAL
    22. QUCO2
    23. QUGE2
    24. QUHE2
    25. QULA2
    26. QULA3
    27. QUMO4
    28. QUNI
    29. QURU
    30. QUERC
    31. ROPS
    32. TSCA

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.7717/peerj.5843

    """

    classes = {
        "ACPE": {"name": "Acer pensylvanicum L."},
        "ACRU": {"name": "Acer rubrum L."},
        "ACSA3": {"name": "Acer saccharum Marshall"},
        "AMLA": {"name": "Amelanchier laevis Wiegand"},
        "BETUL": {"name": "Betula sp."},
        "CAGL8": {"name": "Carya glabra (Mill.) Sweet"},
        "CATO6": {"name": "Carya tomentosa (Lam.) Nutt."},
        "FAGR": {"name": "Fagus grandifolia Ehrh."},
        "GOLA": {"name": "Gordonia lasianthus (L.) Ellis"},
        "LITU": {"name": "Liriodendron tulipifera L."},
        "LYLU3": {"name": "Lyonia lucida (Lam.) K. Koch"},
        "MAGNO": {"name": "Magnolia sp."},
        "NYBI": {"name": "Nyssa biflora Walter"},
        "NYSY": {"name": "Nyssa sylvatica Marshall"},
        "OXYDE": {"name": "Oxydendrum sp."},
        "PEPA37": {"name": "Persea palustris (Raf.) Sarg."},
        "PIEL": {"name": "Pinus elliottii Engelm."},
        "PIPA2": {"name": "Pinus palustris Mill."},
        "PINUS": {"name": "Pinus sp."},
        "PITA": {"name": "Pinus taeda L."},
        "PRSE2": {"name": "Prunus serotina Ehrh."},
        "QUAL": {"name": "Quercus alba L."},
        "QUCO2": {"name": "Quercus coccinea"},
        "QUGE2": {"name": "Quercus geminata Small"},
        "QUHE2": {"name": "Quercus hemisphaerica W. Bartram ex Willd."},
        "QULA2": {"name": "Quercus laevis Walter"},
        "QULA3": {"name": "Quercus laurifolia Michx."},
        "QUMO4": {"name": "Quercus montana Willd."},
        "QUNI": {"name": "Quercus nigra L."},
        "QURU": {"name": "Quercus rubra L."},
        "QUERC": {"name": "Quercus sp."},
        "ROPS": {"name": "Robinia pseudoacacia L."},
        "TSCA": {"name": "Tsuga canadensis (L.) Carriere"},
    }
    metadata = {
        "train": {
            "url": "https://zenodo.org/record/3934932/files/IDTREES_competition_train_v2.zip?download=1",  # noqa: E501
            "md5": "5ddfa76240b4bb6b4a7861d1d31c299c",
            "filename": "IDTREES_competition_train_v2.zip",
        },
        "test": {
            "url": "https://zenodo.org/record/3934932/files/IDTREES_competition_test_v2.zip?download=1",  # noqa: E501
            "md5": "b108931c84a70f2a38a8234290131c9b",
            "filename": "IDTREES_competition_test_v2.zip",
        },
    }
    directories = {
        "train": ["train"],
        "test": ["task1", "task2"],
    }
    data_types = ["rgb", "hsi", "chm", "las"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        task: str = "task1",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new IDTReeS dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            task: 'task1' for detection, 'task2' for detection + classification
                (only relevant for split='test')
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in ["train", "test"]
        assert task in ["task1", "task2"]
        self.root = root
        self.split = split
        self.task = task
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self._verify()
        self.images, self.geometries, self.labels = self._load(root)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        path = self.images[index]

        if self.split == "test":
            if self.task == "task1":
                image = self._load_image(path)
                sample = {"image": image}
            else:
                image = self._load_image(path)
                boxes = self._load_boxes(path)
                sample = {"image": image, "boxes": boxes}
        else:
            image = self._load_image(path)
            hsi = self._load_image(path.replace("RGB", "HSI"))
            chm = self._load_image(path.replace("RGB", "CHM"))
            las = self._load_las(path.replace("RGB", "LAS").replace(".tif", ".las"))
            boxes = self._load_boxes(path)
            label = self._load_target(path)
            sample = {
                "image": image,
                "hsi": hsi,
                "chm": chm,
                "las": las,
                "boxes": boxes,
                "label": label,
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.images)

    def _load_image(self, path: str) -> Tensor:
        """Load a tiff file.

        Args:
            path: path to .tif file

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array = f.read()
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        return tensor

    def _load_las(self, path: str) -> Tensor:
        """Load a single point cloud.

        Args:
            path: path to .las file

        Returns:
            the point cloud
        """
        try:
            import laspy
        except ImportError:
            raise ImportError(
                "laspy is not installed and is required to use this dataset"
            )
        las = laspy.read(path)
        array = np.stack([las.x, las.y, las.z], axis=0)
        tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        return tensor

    def _load_boxes(self, path: str) -> Tensor:
        base_path = os.path.basename(path)

        # Find object ids and geometries
        if self.split == "train":
            indices = self.labels["rsFile"] == base_path
            ids = self.labels[indices]["id"].tolist()
            geoms = [self.geometries[i]["geometry"]["coordinates"][0][:4] for i in ids]
        # Test set - Task 2 has no mapping csv. Mapping is inside of geometry
        else:
            ids = [
                k for k, v in self.geometries if v["properties"]["plotID"] == base_path
            ]
            geoms = [self.geometries[i]["geometry"]["coordinates"][0][:4] for i in ids]

        # Convert to pixel coords
        boxes = []
        with rasterio.open(path) as f:
            for geom in geoms:
                coords = [f.index(x, y) for x, y in geom]
                xmin = min([coord[0] for coord in coords])
                xmax = max([coord[0] for coord in coords])
                ymin = min([coord[1] for coord in coords])
                ymax = max([coord[1] for coord in coords])
                boxes.append([xmin, ymin, xmax, ymax])

        tensor: Tensor = torch.tensor(boxes)  # type: ignore[attr-defined]
        return tensor

    def _load_target(self, path: str) -> Tensor:
        """Load the boxes and target label for a single sample.

        Args:
            path: path to image

        Returns:
            the target boxes (xyxy) and label
        """
        # Find indices for objects in the image
        base_path = os.path.basename(path)
        indices = self.labels["rsFile"] == base_path

        # Load object labels
        classes = self.labels[indices]["taxonID"].tolist()
        labels = [self.class2idx[c] for c in classes]
        tensor: Tensor = torch.tensor(labels)  # type: ignore[attr-defined]
        return tensor

    def _load_labels(self, directory: str) -> pd.DataFrame:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )
        path_mapping = os.path.join(directory, "Field", "itc_rsFile.csv")
        path_labels = os.path.join(directory, "Field", "train_data.csv")
        df_mapping = pd.read_csv(path_mapping)
        df_labels = pd.read_csv(path_labels)
        df_mapping = df_mapping.set_index("indvdID", drop=True)
        df_labels = df_labels.set_index("indvdID", drop=True)
        df = df_labels.join(df_mapping, on="indvdID")
        df = df.drop_duplicates()
        df.reset_index()
        return df

    def _load_geometries(self, directory: str) -> Dict[int, Dict[str, Any]]:
        filepaths = glob.glob(os.path.join(directory, "ITC", "*.shp"))

        features: Dict[int, Dict[str, Any]] = {}
        for path in filepaths:
            with fiona.open(path) as src:
                for feature in src:
                    features[feature["properties"]["id"]] = feature
        return features

    def _load_images(self, directory: str) -> List[str]:
        return glob.glob(os.path.join(directory, "RemoteSensing", "RGB", "*.tif"))

    def _load(
        self, root: str
    ) -> Tuple[List[str], Dict[int, Dict[str, Any]], pd.DataFrame]:
        """Load a files, geometries, and labels.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        if self.split == "train":
            directory = os.path.join(root, self.directories[self.split][0])
            labels = self._load_labels(directory)
            geoms = self._load_geometries(directory)
        else:
            directory = os.path.join(root, self.task)
            if self.task == "task1":
                geoms = None
                labels = None
            else:
                geoms = self._load_geometries(directory)
                labels = None

        images = self._load_images(directory)

        return images, geoms, labels

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        url = self.metadata[self.split]["url"]
        md5 = self.metadata[self.split]["md5"]
        filename = self.metadata[self.split]["filename"]
        directories = self.directories[self.split]

        # Check if the files already exist
        exists = [
            os.path.exists(os.path.join(self.root, directory))
            for directory in directories
        ]
        if all(exists):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, filename)
        if os.path.exists(filepath):
            self._extract(filepath)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download and extract the dataset
        self._download(url, filename, md5)
        filepath = os.path.join(self.root, filename)
        self._extract(filepath)

    def _download(self, url: str, filename: str, md5: str) -> None:
        """Download the dataset.

        Args:
            url: url to download file
            filename: output filename to write downloaded file
            md5: md5 of downloaded file
        """
        download_url(
            url,
            self.root,
            filename=filename,
            md5=md5 if self.checksum else None,
        )

    def _extract(self, filepath: str) -> None:
        """Extract the dataset.

        Args:
            filepath: path to file to be extracted
        """
        extract_archive(filepath)
