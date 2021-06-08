import bz2
import os
import tarfile
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_url,
)


class COWCDetection(VisionDataset):
    """The `Cars Overhead With Context (COWC) <https://gdo152.llnl.gov/cowc/>`_ data set
    is a large set of annotated cars from overhead. It is useful for training a device
    such as a deep neural network to learn to detect and/or count cars.

    The dataset has the following attributes:

    1. Data from overhead at 15 cm per pixel resolution at ground (all data is EO).
    2. Data from six distinct locations: Toronto, Canada; Selwyn, New Zealand; Potsdam
       and Vaihingen, Germany; Columbus, Ohio and Utah, United States.
    3. 32,716 unique annotated cars. 58,247 unique negative examples.
    4. Intentional selection of hard negative examples.
    5. Established baseline for detection and counting tasks.
    6. Extra testing scenes for use after validation.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1007/978-3-319-46487-9_48
    """

    base_folder = "cowc"
    base_url = (
        "https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/"
    )
    filenames = [
        "COWC_train_list_detection.txt.bz2",
        "COWC_test_list_detection.txt.bz2",
        "COWC_Detection_Toronto_ISPRS.tbz",
        "COWC_Detection_Selwyn_LINZ.tbz",
        "COWC_Detection_Potsdam_ISPRS.tbz",
        "COWC_Detection_Vaihingen_ISPRS.tbz",
        "COWC_Detection_Columbus_CSUAV_AFRL.tbz",
        "COWC_Detection_Utah_AGRC.tbz",
    ]
    md5s = [
        "c954a5a3dac08c220b10cfbeec83893c",
        "c6c2d0a78f12a2ad88b286b724a57c1a",
        "11af24f43b198b0f13c8e94814008a48",
        "22fd37a86961010f5d519a7da0e1fc72",
        "bf053545cc1915d8b6597415b746fe48",
        "23945d5b22455450a938382ccc2a8b27",
        "f40522dc97bea41b10117d4a5b946a6f",
        "195da7c9443a939a468c9f232fd86ee3",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transform: Optional[Callable[[Image.Image], Any]] = None,
        target_transform: Optional[Callable[[Dict[str, Any]], Any]] = None,
        transforms: Optional[
            Callable[[Image.Image, Dict[str, Any]], Tuple[Any, Any]]
        ] = None,
        download: bool = False,
    ) -> None:
        """Initialize a new VHR-10 dataset instance.

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

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        assert split in ["train", "test"]

        super().__init__(root, transforms, transform, target_transform)
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = index % len(self) + 1
        image = self._load_image(id_)
        target = self._load_target(id_)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        if self.split == "positive":
            return 650
        else:
            return 150

    def _load_image(self, id_: int) -> Image.Image:
        """Load a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the image
        """
        return Image.open(
            os.path.join(
                self.root,
                self.base_folder,
                "NWPU VHR-10 dataset",
                self.split + " image set",
                f"{id_:03d}.jpg",
            )
        ).convert("RGB")

    def _load_target(self, id_: int) -> Dict[str, Any]:
        """Load the annotations for a single image.

        Parameters:
            id_: unique ID of the image

        Returns:
            the annotations
        """
        # Images in the "negative" image set have no annotations
        annot = []
        if self.split == "positive":
            annot = self.coco.loadAnns(self.coco.getAnnIds(id_))

        target = dict(image_id=id_, annotations=annot)

        return target

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset MD5s match, else False
        """
        for filename, md5 in zip(self.filenames, self.md5s):
            if not check_integrity(
                os.path.join(self.root, self.base_folder, filename), md5
            ):
                return False
        return True

    def download(self) -> None:
        """Download the dataset and extract it."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for filename, md5 in zip(self.filenames, self.md5s):
            download_url(
                self.base_url + filename,
                os.path.join(self.root, self.base_folder),
                filename=filename,
                md5=md5,
            )
            if filename.endswith('.tbz'):
                with tarfile.TarFile(
                    os.path.join(self.root, self.base_folder, filename)
                ) as f:
                    f.extractall(os.path.join(self.root, self.base_folder))
            elif filename.endswith('.bz2'):
                with bz2.BZ2File(os.path.join(self.root, self.base_folder, filename)) as f:

