# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""IDTReeS dataset."""

import os
from typing import Callable, Dict, Optional, Sequence

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

    metadata: Dict[str, Dict[str, str]] = {
        "train": {
            "url": "https://zenodo.org/record/3934932/files/IDTREES_competition_train_v2.zip?download=1",
            "md5": "5ddfa76240b4bb6b4a7861d1d31c299c",
            "filename": "IDTREES_competition_train_v2.zip",
        },
        "test": {
            "url": "https://zenodo.org/record/3934932/files/IDTREES_competition_test_v2.zip?download=1",
            "md5": "b108931c84a70f2a38a8234290131c9b",
            "filename": "IDTREES_competition_test_v2.zip",
        },
    }
    directories: Dict[str, Sequence[str]] = {
        "train": ["train"],
        "test": ["task1", "task2"],
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new IDTReeS dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        self._verify()

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: Dict[str, Tensor] = {
            "image": image,
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
        pass

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        pass

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        pass

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
