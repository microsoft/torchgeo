# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset-specific exceptions."""

from torch.utils.data import Dataset


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a dataset is requested but doesn't exist.

    .. versionadded:: 0.6
    """

    def __init__(self, dataset: Dataset[object]) -> None:
        """Initialize a new DatasetNotFoundError instance.

        Args:
            dataset: The dataset that was requested.
        """
        msg = 'Dataset not found'

        if hasattr(dataset, 'root'):
            var = 'root'
            val = dataset.root
        elif hasattr(dataset, 'paths'):
            var = 'paths'
            val = dataset.paths
        else:
            super().__init__(f'{msg}.')
            return

        msg += f' in `{var}={val!r}` and '

        if hasattr(dataset, 'download') and not dataset.download:
            msg += '`download=False`'
        else:
            msg += 'cannot be automatically downloaded'

        msg += f', either specify a different `{var}` or '

        if hasattr(dataset, 'download') and not dataset.download:
            msg += 'use `download=True` to automatically'
        else:
            msg += 'manually'

        msg += ' download the dataset.'

        super().__init__(msg)


class MissingDependencyError(ModuleNotFoundError):
    """Raised when an optional dataset dependency is not installed.

    .. versionadded:: 0.6
    """

    def __init__(self, name: str) -> None:
        """Initialize a new MissingDependencyError instance.

        Args:
            name: Name of missing dependency.
        """
        msg = f"""\
{name} is not installed and is required to use this dataset. Either run:

$ pip install {name}

to install just this dependency, or:

$ pip install torchgeo[datasets]

to install all optional dataset dependencies."""

        super().__init__(msg)


class RGBBandsMissingError(ValueError):
    """Raised when a dataset is missing RGB bands for plotting.

    .. versionadded:: 0.6
    """

    def __init__(self) -> None:
        """Initialize a new RGBBandsMissingError instance."""
        msg = 'Dataset does not contain some of the RGB bands'
        super().__init__(msg)
