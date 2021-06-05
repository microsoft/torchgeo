import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from .utils import working_dir


class CV4AKenyaCropType(VisionDataset):
    """CV4A Kenya Crop Type dataset.

    Used in a competition in the Computer Vision for Agriculture (CV4A) workshop in ICLR 2020.
    See the competition website <https://zindi.africa/competitions/iclr-workshop-challenge-2-radiant-earth-computer-vision-for-crop-recognition>.

    Consists of 4 tiles of Sentinel 2 imagery from 13 different points in time.

    Each tile has:
    * 13 multi-band observations throughout the growing season. Each observation includes 12 bands from Sentinel-2 L2A product, and a cloud probability layer. The twelve bands are [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] (refer to Sentinel-2 documentation for more information about the bands). The cloud probability layer is a product of the Sentinel-2 atmospheric correction algorithm (Sen2Cor) and provides an estimated cloud probability (0-100%) per pixel. All of the bands are mapped to a common 10 m spatial resolution grid.
    * A raster layer indicating the crop ID for the fields in the training set.
    * A raster layer indicating field IDs for the fields (both training and test sets). Fields with a crop ID 0 are the test fields.

    There are 3,286 fields in the train set and 1,402 fields in the test set.

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
       imagery and labels from the Radiant Earth MLHub
    """

    base_folder = "ref_african_crops_kenya_02"
    image_meta = {
        "filename": "ref_african_crops_kenya_02_source.tar.gz",
        "md5": "9c2004782f6dc83abb1bf45ba4d0da46",
    }
    target_meta = {
        "filename": "ref_african_crops_kenya_02_labels.tar.gz",
        "md5": "93949abd0ae82ba564f5a933cefd8215",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[Image.Image], Any]] = None,
        transforms: Optional[Callable[[Image.Image, Image.Image], Any]] = None,
        download: bool = False,
        api_key: Optional[str] = None
    ) -> None:
        """Initialize a new CV4A Kenya Crop Type dataset instance.

        Parameters:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transform: a function/transform that takes in a PIL image and returns a
                transformed version
            target_transform: a function/transform that takes in the target and
                transforms it
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
        """
        assert split in ["train", "test"]

        super().__init__(root, transforms, transform, target_transform)

        if download:
            if api_key is None:
                raise RuntimeError("You must pass an MLHub API key if download=True. See https://www.mlhub.earth/ to register for API access.")
            else:
                self.download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return an index within the dataset.

        TODO: See the following link for example loading code from competition organizers: https://github.com/radiantearth/mlhub-tutorials/blob/main/notebooks/2020%20CV4A%20Crop%20Type%20Challenge/cv4a-crop-challenge-load-data.ipynb

        Parameters:
            index: index to return

        Returns:
            data and label at that index
        """
        raise NotImplementedError("") # TODO: Implement after discussion about how to handle tile datasets

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        raise NotImplementedError("") # TODO: Implement after discussion about how to handle tile datasets

    def _load_image(self, id_: str) -> Image.Image:
        """
        """
        raise NotImplementedError("") # TODO: Implement after discussion about how to handle tile datasets

    def _load_target(self, id_: str) -> Image.Image:
        """
        """
        raise NotImplementedError("") # TODO: Implement after discussion about how to handle tile datasets

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if the MD5s of the dataset's archives match, else False
        """
        images: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.image_meta["filename"]),
            self.image_meta["md5"],
        )

        targets: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.target_meta["filename"]),
            self.target_meta["md5"],
        )

        return images and targets

    def download(self, api_key) -> None:
        """Download the dataset and extract it.

        Parameters:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
        """

        if self._check_integrity():
            print("Files already downloaded and verified")
            return


        # Download from MLHub and check integrity
        import radiant_mlhub # To download from MLHub, could probably use `requests` instead

        dataset = radiant_mlhub.Dataset.fetch('ref_african_crops_kenya_02', api_key=api_key)
        dataset.download(
            output_dir=os.path.join(self.root, self.base_folder),
            api_key=api_key
        ) # NOTE: Will not work with library versions < 0.2.1

        if not self._check_integrity():
            raise RuntimeError("Dataset files not found or corrupted.")


        # Extract archives
        import tarfile # To extract .tar.gz archives

        image_archive_path = os.path.join(self.root, self.base_folder, self.image_meta["filename"])
        target_archive_path = os.path.join(self.root, self.base_folder, self.target_meta["filename"])
        for fn in [image_archive_path, target_archive_path]:
            with tarfile.open(fn) as tfile:
                tfile.extractall(path=os.path.join(self.root, self.base_folder))