from typing import Any, Callable, Tuple

from torchvision.transforms import Compose, ToTensor

from .transforms import ConvertCocoPolysToMask, RandomHorizontalFlip, RandomVerticalFlip


__all__ = ("Compose", "ConvertCocoPolysToMask", "RandomHorizontalFlip", "RandomVerticalFlip", "ToTensor")


def get_transforms(task: str) -> Tuple[Callable[[Any, Any], Tuple[Any, Any]], Callable[[Any, Any], Tuple[Any, Any]]]:
    """Initialize train and test transforms.

    Parameters:
        task: the task we are trying to perform

    Returns:
        train and test transforms

    Raises:
        ValueError: if ``task`` is not supported
    """
    if task == "detection":
        train_transforms = Compose([
            ToTensor(),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
        ])
        test_transforms = Compose([
            ToTensor(),
        ])
    elif task == "segmentation":
        train_transforms = Compose([
            ConvertCocoPolysToMask(),
            ToTensor(),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
        ])
        test_transforms = Compose([
            ConvertCocoPolysToMask(),
            ToTensor(),
        ])
    else:
        raise ValueError(f"Unsupported task: '{task}'")

    return train_transforms, test_transforms
