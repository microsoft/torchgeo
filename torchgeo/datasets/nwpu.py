import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    download_url,
)


class VHR10(VisionDataset):
    """`NWPU VHR-10 <https://doi.org/10.1016/j.isprsjprs.2014.10.002>`_ Dataset."""

    base_folder = "vhr10"
    image_meta = {
        "file_id": "1--foZ3dV5OCsqXQXT84UeKtrAqc5CkAE",
        "filename": "NWPU VHR-10 dataset.rar",
        "md5": "d30a7ff99d92123ebb0b3a14d9102081",
    }
    target_meta = {
        "url": (
            "https://raw.githubusercontent.com/chaozhong2010/VHR-10_dataset_coco/"
            "master/NWPU%20VHR-10_dataset_coco/annotations.json"
        ),
        "filename": "annotations.json",
        "md5": "7c76ec50c17a61bb0514050d20f22c08",
    }

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
        transforms: Optional[Callable[[Any], Any]] = None,
        download: bool = False,
    ) -> None:
        """Initialize a new VHR-10 dataset instance.

        Parameters:
            root: root directory where dataset can be found
            transform: a function/transform that takes in a PIL image and returns a
                transformed version
            target_transform: a function/transform that takes in the target and
                transforms it
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
        """
        super().__init__(root, transforms, transform, target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        # Must be installed to parse annotations file
        from pycocotools.coco import COCO

        self.coco = COCO(
            os.path.join(
                self.root,
                self.base_folder,
                "NWPU VHR-10 dataset",
                self.target_meta["filename"],
            )
        )
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return an index within the dataset.

        Parameters:
            idx: index to return

        Returns:
            data and label at that index
        """
        id = self.ids[index]
        image = self._load_image(id)
        annot = self._load_target(id)

        target = dict(image_id=id, annotations=annot)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_image(self, id: int) -> Image.Image:
        """Load a single image.

        Parameters:
            id: unique ID of the image

        Returns:
            the image
        """
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(
            os.path.join(
                self.root,
                self.base_folder,
                "NWPU VHR-10 dataset",
                "positive image set",
                path,
            )
        ).convert("RGB")

    def _load_target(self, id: int) -> Any:
        """Load the annotations for a single image.

        Parameters:
            id: unique ID of the image

        Returns:
            the annotations
        """
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset MD5s match, else False
        """
        image: bool = check_integrity(
            os.path.join(self.root, self.base_folder, self.image_meta["filename"]),
            self.image_meta["md5"],
        )
        target: bool = check_integrity(
            os.path.join(
                self.root,
                self.base_folder,
                "NWPU VHR-10 dataset",
                self.target_meta["filename"],
            ),
            self.target_meta["md5"],
        )
        return image and target

    def download(self) -> None:
        """Download the dataset and extract it."""

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_file_from_google_drive(
            self.image_meta["file_id"],
            os.path.join(self.root, self.base_folder),
            self.image_meta["filename"],
            self.image_meta["md5"],
        )

        # Must be installed to extract RAR file
        import rarfile

        with rarfile.RarFile(
            os.path.join(self.root, self.base_folder, self.image_meta["filename"])
        ) as f:
            f.extractall(os.path.join(self.root, self.base_folder))

        download_url(
            self.target_meta["url"],
            os.path.join(self.root, self.base_folder, "NWPU VHR-10 dataset"),
            self.target_meta["filename"],
            self.target_meta["md5"],
        )
