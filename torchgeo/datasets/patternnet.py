# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PatternNet dataset."""

import os
from typing import Callable, Dict, Optional, cast

import matplotlib.pyplot as plt
from torch import Tensor

from .geo import NonGeoClassificationDataset
from .utils import download_url, extract_archive


class PatternNet(NonGeoClassificationDataset):
    """PatternNet dataset.

    The `PatternNet <https://sites.google.com/view/zhouwx/dataset>`__
    dataset is a dataset for remote sensing scene classification and image retrieval.

    Dataset features:

    * 30,400 images with 6-50 cm per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 38 scene classes, 800 images per class

    Dataset format:

    * images are three-channel jpgs

    Dataset classes:

    0. airplane
    1. baseball_field
    2. basketball_court
    3. beach
    4. bridge
    5. cemetery
    6. chaparral
    7. christmas_tree_farm
    8. closed_road
    9. coastal_mansion
    10. crosswalk
    11. dense_residential
    12. ferry_terminal
    13. football_field
    14. forest
    15. freeway
    16. golf_course
    17. harbor
    18. intersection
    19. mobile_home_park
    20. nursing_home
    21. oil_gas_field
    22. oil_well
    23. overpass
    24. parking_lot
    25. parking_space
    26. railway
    27. river
    28. runway
    29. runway_marking
    30. shipping_yard
    31. solar_panel
    32. sparse_residential
    33. storage_tank
    34. swimming_pool
    35. tennis_court
    36. transformer_station
    37. wastewater_treatment_plant

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.isprsjprs.2018.01.004
    """

    url = "https://drive.google.com/file/d/127lxXYqzO6Bd0yZhvEbgIfz95HaEnr9K"
    md5 = "96d54b3224c5350a98d55d5a7e6984ad"
    filename = "PatternNet.zip"
    directory = os.path.join("PatternNet", "images")

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new PatternNet dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        self.root = root
        self.download = download
        self.checksum = checksum
        self._verify()
        super().__init__(root=os.path.join(root, self.directory), transforms=transforms)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.directory)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image, label = sample["image"], cast(int, sample["label"].item())

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(int, sample["prediction"].item())

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image.permute(1, 2, 0))
        ax.axis("off")

        if show_titles:
            title = f"Label: {self.classes[label]}"
            if showing_predictions:
                title += f"\nPrediction: {self.classes[prediction]}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
