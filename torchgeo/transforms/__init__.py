from .transforms import Identity, RandomHorizontalFlip, RandomVerticalFlip

# https://stackoverflow.com/questions/40018681
Identity.__module__ = "torchgeo.transforms"
RandomHorizontalFlip.__module__ = "torchgeo.transforms"
RandomVerticalFlip.__module__ = "torchgeo.transforms"


__all__ = ("Identity", "RandomHorizontalFlip", "RandomVerticalFlip")
