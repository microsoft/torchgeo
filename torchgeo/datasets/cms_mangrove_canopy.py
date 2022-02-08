# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CMS Global Mangrove Canopy dataset."""

import abc
import os
from typing import Any, Callable, Dict, Optional

from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import check_integrity


class CMS_Global_Mangrove_Canopy(RasterDataset, abc.ABC):
    """CMS Global Mangrove Canopy dataset.

    The 'CMS Global Mangrove Canopy dataset
    <https://daac.ornl.gov/cgi-bin/dsviewer.pl?ds_id=1665>'
    consists of a single band map at 30m resolution of either aboveground biomass (agb),
    basal area weighted height (hba95), or maximum canopy height (hmax95).

    The dataset needs to be manually dowloaded from , where you can make an account
    and subsequently download the dataset.
    """

    is_image = False

    filename_regex = r"""^
        (?P<mangrove>[A-Za-z]{8})
        _(?P<variable>[a-z0-9]*)
        _(?P<country>[A-Za-z][^.]*)
    """

    zipfile = "CMS_Global_Map_Mangrove_Canopy_1665.zip"
    md5 = "3e7f9f23bf971c25e828b36e6c5496e3"

    all_countries = [
        "AndamanAndNicobar",
        "Angola",
        "Anguilla",
        "AntiguaAndBarbuda",
        "Aruba",
        "Australia",
        "Bahamas",
        "Bahrain",
        "Bangladesh",
        "Barbados",
        "Belize",
        "Benin",
        "Brazil",
        "BritishVirginIslands",
        "Brunei",
        "Cambodia",
        "Cameroon",
        "CarribeanCaymanIslands",
        "China",
        "Colombia",
        "Comoros",
        "CostaRica",
        "Cote",
        "CoteDivoire",
        "CotedIvoire",
        "Cuba",
        "DemocraticRepublicOfCongo",
        "Djibouti",
        "DominicanRepublic",
        "EcuadorWithGalapagos",
        "Egypt",
        "ElSalvador",
        "EquatorialGuinea",
        "Eritrea",
        "EuropaIsland",
        "Fiji",
        "Fiji2",
        "FrenchGuiana",
        "FrenchGuyana",
        "FrenchPolynesia",
        "Gabon",
        "Gambia",
        "Ghana",
        "Grenada",
        "Guadeloupe",
        "Guam",
        "Guatemala",
        "Guinea",
        "GuineaBissau",
        "Guyana",
        "Haiti",
        "Hawaii",
        "Honduras",
        "HongKong",
        "India",
        "Indonesia",
        "Iran",
        "Jamaica",
        "Japan",
        "Kenya",
        "Liberia",
        "Macau",
        "Madagascar",
        "Malaysia",
        "Martinique",
        "Mauritania",
        "Mayotte",
        "Mexico",
        "Micronesia",
        "Mozambique",
        "Myanmar",
        "NewCaledonia",
        "NewZealand",
        "Newzealand",
        "Nicaragua",
        "Nigeria",
        "NorthernMarianaIslands",
        "Oman",
        "Pakistan",
        "Palau",
        "Panama",
        "PapuaNewGuinea",
        "Peru",
        "Philipines",
        "PuertoRico",
        "Qatar",
        "ReunionAndMauritius",
        "SaintKittsAndNevis",
        "SaintLucia",
        "SaintVincentAndTheGrenadines",
        "Samoa",
        "SaudiArabia",
        "Senegal",
        "Seychelles",
        "SierraLeone",
        "Singapore",
        "SolomonIslands",
        "Somalia",
        "Somalia2",
        "Soudan",
        "SouthAfrica",
        "SriLanka",
        "Sudan",
        "Suriname",
        "Taiwan",
        "Tanzania",
        "Thailand",
        "TimorLeste",
        "Togo",
        "Tonga",
        "TrinidadAndTobago",
        "TurksAndCaicosIslands",
        "Tuvalu",
        "UnitedArabEmirates",
        "UnitedStates",
        "Vanuatu",
        "Venezuela",
        "Vietnam",
        "VirginIslandsUs",
        "WallisAndFutuna",
        "Yemen",
    ]

    measurements = ["agb", "hba95", "hmax95"]

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        measurement: str = "agb",
        country: str = all_countries[0],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
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
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.checksum = checksum

        assert isinstance(country, str), "Country argument must be a str."
        assert (
            country in self.all_countries
        ), "You have selected an invalid country, please choose one of {}".format(
            self.all_countries
        )
        self.country = country

        assert isinstance(measurement, str), "Measurement must be a string."
        assert (
            measurement in self.measurements
        ), "You have entered an invalid measurement, please choose one of {}.".format(
            self.measurements
        )
        self.measurement = measurement

        self.filename_glob = "Mangrove_{}_{}*".format(self.measurement, self.country)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                "Please ensure you have manually downloaded it."
            )

        super().__init__(root, crs, res, transforms, cache)

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.zipfile), self.md5 if self.checksum else None
        )

        return integrity
