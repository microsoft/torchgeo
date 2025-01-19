# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CMS Global Mangrove Canopy dataset."""

import os
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import Path, check_integrity, extract_archive


class CMSGlobalMangroveCanopy(RasterDataset):
    """CMS Global Mangrove Canopy dataset.

    The `CMS Global Mangrove Canopy dataset
    <https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1665>`_
    consists of a single band map at 30m resolution of either aboveground biomass (agb),
    basal area weighted height (hba95), or maximum canopy height (hmax95).

    The dataset needs to be manually downloaded from the above link, where you can make
    an account and subsequently download the dataset.

    .. versionadded:: 0.3
    """

    is_image = False

    filename_regex = r"""^
        (?P<mangrove>[A-Za-z]{8})
        _(?P<variable>[a-z0-9]*)
        _(?P<country>[A-Za-z][^.]*)
    """

    zipfile = 'CMS_Global_Map_Mangrove_Canopy_1665.zip'
    md5 = '3e7f9f23bf971c25e828b36e6c5496e3'

    all_countries = (
        'AndamanAndNicobar',
        'Angola',
        'Anguilla',
        'AntiguaAndBarbuda',
        'Aruba',
        'Australia',
        'Bahamas',
        'Bahrain',
        'Bangladesh',
        'Barbados',
        'Belize',
        'Benin',
        'Brazil',
        'BritishVirginIslands',
        'Brunei',
        'Cambodia',
        'Cameroon',
        'CarribeanCaymanIslands',
        'China',
        'Colombia',
        'Comoros',
        'CostaRica',
        'Cote',
        'CoteDivoire',
        'CotedIvoire',
        'Cuba',
        'DemocraticRepublicOfCongo',
        'Djibouti',
        'DominicanRepublic',
        'EcuadorWithGalapagos',
        'Egypt',
        'ElSalvador',
        'EquatorialGuinea',
        'Eritrea',
        'EuropaIsland',
        'Fiji',
        'Fiji2',
        'FrenchGuiana',
        'FrenchGuyana',
        'FrenchPolynesia',
        'Gabon',
        'Gambia',
        'Ghana',
        'Grenada',
        'Guadeloupe',
        'Guam',
        'Guatemala',
        'Guinea',
        'GuineaBissau',
        'Guyana',
        'Haiti',
        'Hawaii',
        'Honduras',
        'HongKong',
        'India',
        'Indonesia',
        'Iran',
        'Jamaica',
        'Japan',
        'Kenya',
        'Liberia',
        'Macau',
        'Madagascar',
        'Malaysia',
        'Martinique',
        'Mauritania',
        'Mayotte',
        'Mexico',
        'Micronesia',
        'Mozambique',
        'Myanmar',
        'NewCaledonia',
        'NewZealand',
        'Newzealand',
        'Nicaragua',
        'Nigeria',
        'NorthernMarianaIslands',
        'Oman',
        'Pakistan',
        'Palau',
        'Panama',
        'PapuaNewGuinea',
        'Peru',
        'Philipines',
        'PuertoRico',
        'Qatar',
        'ReunionAndMauritius',
        'SaintKittsAndNevis',
        'SaintLucia',
        'SaintVincentAndTheGrenadines',
        'Samoa',
        'SaudiArabia',
        'Senegal',
        'Seychelles',
        'SierraLeone',
        'Singapore',
        'SolomonIslands',
        'Somalia',
        'Somalia2',
        'Soudan',
        'SouthAfrica',
        'SriLanka',
        'Sudan',
        'Suriname',
        'Taiwan',
        'Tanzania',
        'Thailand',
        'TimorLeste',
        'Togo',
        'Tonga',
        'TrinidadAndTobago',
        'TurksAndCaicosIslands',
        'Tuvalu',
        'UnitedArabEmirates',
        'UnitedStates',
        'Vanuatu',
        'Venezuela',
        'Vietnam',
        'VirginIslandsUs',
        'WallisAndFutuna',
        'Yemen',
    )

    measurements = ('agb', 'hba95', 'hmax95')

    def __init__(
        self,
        paths: Path | list[Path] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        measurement: str = 'agb',
        country: str = all_countries[0],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            measurement: which of the three measurements, 'agb', 'hba95', or 'hmax95'
            country: country for which to retrieve data
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if country or measurement arg are not str or invalid
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.paths = paths
        self.checksum = checksum

        assert isinstance(country, str), 'Country argument must be a str.'
        assert country in self.all_countries, (
            f'You have selected an invalid country, please choose one of {self.all_countries}'
        )
        self.country = country

        assert isinstance(measurement, str), 'Measurement must be a string.'
        assert measurement in self.measurements, (
            f'You have entered an invalid measurement, please choose one of {self.measurements}.'
        )
        self.measurement = measurement

        self.filename_glob = f'**/Mangrove_{self.measurement}_{self.country}*'

        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        if self.files:
            return

        # Check if the zip file has already been downloaded
        assert isinstance(self.paths, str | os.PathLike)
        pathname = os.path.join(self.paths, self.zipfile)
        if os.path.exists(pathname):
            if self.checksum and not check_integrity(pathname, self.md5):
                raise RuntimeError('Dataset found, but corrupted.')
            self._extract()
            return

        raise DatasetNotFoundError(self)

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        pathname = os.path.join(self.paths, self.zipfile)
        extract_archive(pathname)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample['mask'].squeeze()
        ncols = 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = sample['prediction'].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis('off')
            axs[1].imshow(pred)
            axs[1].axis('off')
            if show_titles:
                axs[0].set_title('Mask')
                axs[1].set_title('Prediction')
        else:
            axs.imshow(mask)
            axs.axis('off')
            if show_titles:
                axs.set_title('Mask')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
