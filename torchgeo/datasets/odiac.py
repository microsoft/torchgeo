# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ODIAC Fossil Fuel CO2 Emissions dataset."""

import gzip
import os
import re
import shutil
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, cast
from pathlib import Path as PathlibPath

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from pyproj import Transformer
from rtree.index import Index, Property
from rasterio.crs import CRS
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import GeoDataset, RasterDataset
from .utils import BoundingBox, Path, disambiguate_timestamp, download_url


class ODIAC(RasterDataset):
    """ODIAC Fossil Fuel CO2 Emissions dataset.

    The `Open-Data Inventory for Anthropogenic Carbon dioxide (ODIAC)
    <https://db.cger.nies.go.jp/dataset/ODIAC/>`_ is a high-spatial resolution
    (1x1km) global emission data product of CO2 emissions from fossil fuel
    combustion. This dataset provides monthly emissions.

    Dataset features:

    * Global CO2 emissions from fossil fuel combustion
    * 1x1 km spatial resolution (approx 0.00833 degrees)
    * Monthly temporal resolution
    * Data spans from 2000 to (Version Year - 1) e.g. ODIAC2023 has data up to 2022.
    * Units are tonnes C yr-1 cell-1 (divide by 1000 for kt C yr-1 grid-1)

    Dataset format:

    * Emissions are single-channel GeoTIFF files (`.tif`), one file per month.
    * Files are downloaded as compressed gzip archives (`.tif.gz`).
    * Files are organized into yearly folders within the root directory
      (e.g. `<root>/<year>/odiac<VERSION>_1km_excl_intl_<YYMM>.tif`).

    If you use this dataset in your research, please cite the following papers:

    * Oda, T., & Maksyutov, S. (2011). A very high-resolution (1 km× 1 km) global inventory
      of anthropogenic carbon dioxide emissions from fossil fuel consumption.
      Atmospheric Chemistry and Physics, 11(2), 543-556.
      https://doi.org/10.5194/acp-11-543-2011
    * Oda, T., Maksyutov, S., & Andres, R. J. (2018). The Open-source Data Inventory for
      Anthropogenic CO2, version 2016 (ODIAC2016): a global monthly fossil fuel CO2
      gridded emissions data product for tracer transport simulations and surface flux
      inversions. Earth System Science Data, 10(1), 87-107.
      https://doi.org/10.5194/essd-10-87-2018

    .. versionadded:: 0.8
    """

    is_image = True  # Treat emissions as continuous data
    separate_files = True  # One file per month effectively acts like separate files
    all_bands = ('co2_emission',)
    # No standard RGB interpretation, use sequential colormap like magma or inferno
    #cmap = 'magma'

    # Regex for the final extracted .tif file
    filename_regex = r'^odiac(?P<version>\d+)_1km_excl_intl_(?P<yymm>\d{4})\.tif$'
    # Glob matches the final extracted .tif files within yearly directories
    filename_glob = 'odiac*_1km_excl_intl_????.tif' # ? wildcard matches single char

    # Base URL pattern for the *monthly* .tif.gz file
    url = "https://db.cger.nies.go.jp/nies_data/10.17595/20170411.001/odiac{version}/1km_tiff/{year}/odiac{version}_1km_excl_intl_{yymm}.tif.gz" # noqa: E501

    # MD5 checksums for the *monthly .tif.gz files*, keyed by (version, year, month)
    md5s: ClassVar[dict[tuple[int, int, int], str]] = {
        # --- ODIAC2023 ---
        # Year 2022
        (2023, 2022, 1): "74d22d1eb707f1a8509f29667a376d33",
        (2023, 2022, 2): "c5dc53d76b5e2bb8853526a75962a9bf",
        (2023, 2022, 3): "1258c38a6e54ab6db8c80642450798eb",
        (2023, 2022, 4): "1e4c9b75d2ea6220f9ee138c9376174e",
        (2023, 2022, 5): "9b17bda9724c8134cc4a73e693407015",
        (2023, 2022, 6): "d3c00b5dc7de1874ee6e9c0b857af953",
        (2023, 2022, 7): "9e45a26f34ad4f6076204df05f396f01",
        (2023, 2022, 8): "7052c7b39fe089955ec8e43d58fae8a6",
        (2023, 2022, 9): "90f6552796575b8b967870f02a9d1bf5",
        (2023, 2022, 10): "882eb95bbccfa7042a65c8d17250b37a",
        (2023, 2022, 11): "09d27bfe87f0a37104effb4b09e15dd1",
        (2023, 2022, 12): "1b3ac4a6c9c9e95c2ca0706b14584c6e",
        # Year 2021
        (2023, 2021, 1): "0b1b93b1b470f48833667dd9b6647300",
        (2023, 2021, 2): "238743c7b02711dc6ae5a2994c9a72f5",
        (2023, 2021, 3): "c3dc4533e9f23b16a7dcfbb3435309f3",
        (2023, 2021, 4): "0f483fca857b28f2ec57f071ad1602a0",
        (2023, 2021, 5): "689388aa0e3d71322e0c736a4d317929",
        (2023, 2021, 6): "48bc94d858cabbf93d75b39572c284f9",
        (2023, 2021, 7): "f0922b3ce34e730e0f667f8f0d769129",
        (2023, 2021, 8): "1698e2397282326cbe554bc84d74f20e",
        (2023, 2021, 9): "4f6cb6ead8c8033b1da9abaf6a7db044",
        (2023, 2021, 10): "5bcf087318657d99c86f5178f6e69585",
        (2023, 2021, 11): "a9787c78bf69d2cb99639617aaac53f3",
        (2023, 2021, 12): "08d4939662c273c99ab355494806bacd",
        # Year 2020
        (2023, 2020, 1): "a83024aa59a30c5ee75456d2d5f8eb46",
        (2023, 2020, 2): "20133273fa27ead4dd999b0dfb818cc3",
        (2023, 2020, 3): "6dc28e84051384aade5178d0b75eb51b",
        (2023, 2020, 4): "4c30f80ee18a9adab2f7f8510c525b7d",
        (2023, 2020, 5): "66da61a22c0d6ce6a6c45bff54a66a07",
        (2023, 2020, 6): "553c013f4bd057435e3e0ce71629726c",
        (2023, 2020, 7): "97de4638bd235301fad5f7ff454ffb66",
        (2023, 2020, 8): "934b2d8e9661fb7d4028115134b26d44",
        (2023, 2020, 9): "5f092483b07840ba3e367e0f54400415",
        (2023, 2020, 10): "12751fdb32cbe8a6fe435bc9595642d4",
        (2023, 2020, 11): "f17ec9f3635c085bcb384ab46ff9d523",
        (2023, 2020, 12): "e60a8aebd1ae898dd1f3b5f403a9e6f7",
        # Year 2019
        (2023, 2019, 1): "61df259056da06529abcc506b0bd96a9",
        (2023, 2019, 2): "ac32d699048b60ca3ee3cf1e593aecec",
        (2023, 2019, 3): "66bd98ac6ac262d08d52722cdd279022",
        (2023, 2019, 4): "c0e787217735062584e271669ea0040f",
        (2023, 2019, 5): "5ac90bbf5d31d25a45511a75a08904f3",
        (2023, 2019, 6): "402632e4f3a48ea993808a0f344fce22",
        (2023, 2019, 7): "bff6685b95ddc95f254b9cc1ff050996",
        (2023, 2019, 8): "7f36d0a7867844994afd57495ce1d4bc",
        (2023, 2019, 9): "aa43b62ff4e65f9da1f3ccb77ed918f3",
        (2023, 2019, 10): "e9609da43915fce933182fba34426f00",
        (2023, 2019, 11): "4301a26e137a6b9f0c06c88b4127b99f",
        (2023, 2019, 12): "06a99130f5aaa6a2ff651a9b9c15852c",
        # Year 2018
        (2023, 2018, 1): "75187fd8a44dcb9a8b2c79ac23e4dce2",
        (2023, 2018, 2): "3cc5fb2ac3cfd03d4f67db622042815e",
        (2023, 2018, 3): "4f4db15e142d8aa35307695914a66085",
        (2023, 2018, 4): "862d0cc4f486c98006094ac88757953c",
        (2023, 2018, 5): "6e7b1fea5c4f01e022f97a5ce9c53813",
        (2023, 2018, 6): "8cbea0876ede81ff3f65edc23bb31c0c",
        (2023, 2018, 7): "ae7554e35a9e0a998f563606aaa4eec0",
        (2023, 2018, 8): "4c6c1849a41f8aa2497bf84998dfa7cd",
        (2023, 2018, 9): "a3db4288b65b597ab636dc406ac86175",
        (2023, 2018, 10): "aa9d30bfce7cfbcef936c7c2ae544349",
        (2023, 2018, 11): "810118915069fd49123ef729ecf15a4c",
        (2023, 2018, 12): "ecbcf7a879d3cc2cad7b04f50913ee81",
        # Year 2017
        (2023, 2017, 1): "63b907f80a957e107057b8227e84e0b7",
        (2023, 2017, 2): "5599042cbe9b7858c18b4a729355c46e",
        (2023, 2017, 3): "a208dab0fd975f87d95ac71326167ad3",
        (2023, 2017, 4): "81d7e6754e38f8ca10bdb66440994879",
        (2023, 2017, 5): "435ea42888725d531f1da0e0e6b008c8",
        (2023, 2017, 6): "5fb6b66fc755311b8e04f4a41bc23cfc",
        (2023, 2017, 7): "67081858a0704f62bf646a3157bce414",
        (2023, 2017, 8): "a2840f1a1cc43cd44a3ca5af5e4a0aa6",
        (2023, 2017, 9): "9b1996589b9f41f5be143875f43f3b10",
        (2023, 2017, 10): "e0fab83f390ae4c89bd0cc850b3f40e6",
        (2023, 2017, 11): "999cc3675bd1848ac012b2ec160747f3",
        (2023, 2017, 12): "14f0e6bd8f9504e259c61494e2000310",
        # Year 2016
        (2023, 2016, 1): "5ee219aa9045ab8663636e719bac1a26",
        (2023, 2016, 2): "f91366ef485122bc499712ae67da95f4",
        (2023, 2016, 3): "8c556c7e7eb386cd1eca698ef105308f",
        (2023, 2016, 4): "f63f59691b6d4ba541b35436ffb857a0",
        (2023, 2016, 5): "89c8943cb995c30717bc4b5f6023f01a",
        (2023, 2016, 6): "1c1fa8e98d42d85db7b2556c6e99f855",
        (2023, 2016, 7): "d107fe9aff2b35ab5be1531ef3c444de",
        (2023, 2016, 8): "ad4bb7761fd7cf5459548ef42f5e7994",
        (2023, 2016, 9): "4f22e93db3d52309acb9df23e6be2b5e",
        (2023, 2016, 10): "6715a601866d9e84c562357d24716ee7",
        (2023, 2016, 11): "3745c75cd79cee6a54c959f85c3e23c1",
        (2023, 2016, 12): "426ad3291bfc2911391a53c5ddeb7788",
        # Year 2015
        (2023, 2015, 1): "2cf279c1f0b630e32eec2127dba50322",
        (2023, 2015, 2): "4049888b4e6add0d56d3345b8e6964ab",
        (2023, 2015, 3): "b3a60f573c6f018885894964a8d989c8",
        (2023, 2015, 4): "ae7b141b1a3145653346b1d544a1a26f",
        (2023, 2015, 5): "28e6d5e1531a9e991277f08ac6acaa99",
        (2023, 2015, 6): "6bcf3fb5fa819b57846418923fde6346",
        (2023, 2015, 7): "d48e903de88267e564777f0e0ea3c8db",
        (2023, 2015, 8): "b8deb2b285268f58386bc4361baacaa6",
        (2023, 2015, 9): "376f636a8bac9087879b0d96ad9ea559",
        (2023, 2015, 10): "35ba2358a7c4c08c9f844053d4df7bd3",
        (2023, 2015, 11): "4f1adebba39dc76cb6b2b1f5f53e3431",
        (2023, 2015, 12): "7eb1fc5a5ec57af6553467bc98d87e2f",
        # Year 2014
        (2023, 2014, 1): "6f315d55a41ed4874409ecff673e08dd",
        (2023, 2014, 2): "183b1158ce7052adcd8520d64411e42f",
        (2023, 2014, 3): "4bd72cf6e0fb159e17100ea0e802ea49",
        (2023, 2014, 4): "892c004c485ecd818936cae0cf29cf29",
        (2023, 2014, 5): "706f38cce435e48ee7dde1ff80a35b13",
        (2023, 2014, 6): "b96002baeff2d5c849e72bf76417a3f7",
        (2023, 2014, 7): "cf12a9b3d17c62fd026c618496889445",
        (2023, 2014, 8): "2291dd345cf9a0ac5fcd59c49abd8024",
        (2023, 2014, 9): "0cdd0c12a47f4278e9935b0c86e7ee82",
        (2023, 2014, 10): "bca07098dd854e23a36f4e1b0cfeb3d3",
        (2023, 2014, 11): "468d65178ed0e62d3a5ccbfff3a7d343",
        (2023, 2014, 12): "14f584e83d8b622249169a32540e850e",
        # Year 2013
        (2023, 2013, 1): "0956def36344c81bb9bbffbce8ec3ca9",
        (2023, 2013, 2): "becec19cca9b4d64bf37c7e13577d6da",
        (2023, 2013, 3): "9c3febcdc0b346c89718c37e9ce3e798",
        (2023, 2013, 4): "607536b56e1f4937dc6865eb56dccfa4",
        (2023, 2013, 5): "2125b7fbe0510ccb39a2e26183a5907a",
        (2023, 2013, 6): "24bc57673e7d391a0da86de67ed396eb",
        (2023, 2013, 7): "8f4c42b8205485774661834564081609",
        (2023, 2013, 8): "de2d27511d6c1eeee938bfdaa704bb10",
        (2023, 2013, 9): "1bb48cdc5cd964fdc111fb1da67c6264",
        (2023, 2013, 10): "7b61fb0f5506b1eb4a6dcc576a26e0ad",
        (2023, 2013, 11): "5dfd586b7b0e045e8f63ce75c47d6aab",
        (2023, 2013, 12): "6548a5cf15d0d81f2f424d43900ed098",
        # Year 2012
        (2023, 2012, 1): "a581cf703045f132e64aed54225984a0",
        (2023, 2012, 2): "11536874dcf195416a8d95e6bd051abc",
        (2023, 2012, 3): "1e0dcd583cd5cea5aff954db596eec9a",
        (2023, 2012, 4): "779636c78c58b1a8a74061fb74bc9e4c",
        (2023, 2012, 5): "5ca7ded6917fabe06a92d44dbe71046a",
        (2023, 2012, 6): "f9446d1b02420a494604dd7f86583c17",
        (2023, 2012, 7): "6d079dd34b5c6fe2cc517ddc73f9b7ea",
        (2023, 2012, 8): "ed405d45d6174f0e89a214a9aa3f2589",
        (2023, 2012, 9): "b5fa8e8725a8084dde1bf60f709f580a",
        (2023, 2012, 10): "91e24d3a02d29f8970043e718f360fa7",
        (2023, 2012, 11): "34ffd83e92e098462402e572e319f8ac",
        (2023, 2012, 12): "ba7646c2d515f9cc0bcf2988874ba588",
        # Year 2011
        (2023, 2011, 1): "f3908f0ce742248aca20fb47f45f0c38",
        (2023, 2011, 2): "69196630f46c84d13adb064b588ed078",
        (2023, 2011, 3): "c465cc870a553c5d08999ad44c217995",
        (2023, 2011, 4): "77e5255fdabe16fd95cef3cd31ba295a",
        (2023, 2011, 5): "4a0d0e4c0e24d8cee0fa5371a4299ba4",
        (2023, 2011, 6): "12936bfef3369c84637c0a9a9a152145",
        (2023, 2011, 7): "e111f10b3582b126c5ef8323591e163b",
        (2023, 2011, 8): "641581bb3eb82ca67cd13f254dbed9f2",
        (2023, 2011, 9): "6ca922727be7036419589dbf4645a8ff",
        (2023, 2011, 10): "c24a549380364b4039013b9aa18c99d5",
        (2023, 2011, 11): "11990a44945e26a4f0e3b638caf1390e",
        (2023, 2011, 12): "1b525d3921bd487d47999694099ee4c7",
        # Year 2010
        (2023, 2010, 1): "276ada02554cc3996328f290d6acb076",
        (2023, 2010, 2): "21a08b9ebcad2b50ae6e9139f5a1326a",
        (2023, 2010, 3): "156994698342660010f8993e6ee56701",
        (2023, 2010, 4): "f2b33f753c56c871a6d22c166c953a28",
        (2023, 2010, 5): "61dba11751158afa847c3216a67cabd1",
        (2023, 2010, 6): "d108bed20f5a4a07fef98b02043cd1cc",
        (2023, 2010, 7): "f72dba61ad66e08ef68afed5f0dedf55",
        (2023, 2010, 8): "98c03de10664f86c44c74fc00b9ffd53",
        (2023, 2010, 9): "60ca8e800f5aa1f0374ec89c7587db1d",
        (2023, 2010, 10): "e5a6519a6aeb5fc9d073a0c23ea110c0",
        (2023, 2010, 11): "2fc15155e48b4a2f9278d012f94d0bdc",
        (2023, 2010, 12): "5c37ecaeee12b97431ec4cd9525b1af4",
        # Year 2009
        (2023, 2009, 1): "2b4e2ead07ab238021abcbfbea594536",
        (2023, 2009, 2): "e17f48db1a60c186535fc2704c6da3a0",
        (2023, 2009, 3): "51af44f352155168d933b3d31d5bc2b6",
        (2023, 2009, 4): "73c19d14405690e80667a8682ce20d1a",
        (2023, 2009, 5): "6fcad9cddab5f048c3ef31b16a33cd6c",
        (2023, 2009, 6): "afa7544d564c9e2980d52fc0a2e5b499",
        (2023, 2009, 7): "1bf71fa95626fa709377ca813f91f256",
        (2023, 2009, 8): "63ded43d2a0b4910e9ad61b6208730f4",
        (2023, 2009, 9): "d8ec6135d8ec40766b31276b52006803",
        (2023, 2009, 10): "3f1a298138dec1ea157fc65558064b88",
        (2023, 2009, 11): "9903440bb4d06a45f829417452add21b",
        (2023, 2009, 12): "1ddc863a93c754438fcce7c2b9b1e489",
        # Year 2008
        (2023, 2008, 1): "1a4402bb100631359c545f693911fafd",
        (2023, 2008, 2): "47099b4939a3e1e4ccd112832458101c",
        (2023, 2008, 3): "f1a03db9f33b5ad3191a80c6c6f68747",
        (2023, 2008, 4): "9fed34952167ea5d7131a8e3f5470753",
        (2023, 2008, 5): "a3f64eba509259f26619471d18f1c1f5",
        (2023, 2008, 6): "414add19f3d15839699eecf64ed27c9c",
        (2023, 2008, 7): "2fb776f34d38d9d3e8b5f83e2171897d",
        (2023, 2008, 8): "fa57f9605127512361deeb3cf244bea1",
        (2023, 2008, 9): "bb2a7cdc468b865b3bd8ec0e4844164d",
        (2023, 2008, 10): "becaee8077d52adf33773426a3fc5c63",
        (2023, 2008, 11): "1bee69239c954c0775a54c0581b19521",
        (2023, 2008, 12): "1bd1ca7d28e0a23fa4f6076433f44620",
        # Year 2007
        (2023, 2007, 1): "5c599cf1917dbc86daab6109d832a877",
        (2023, 2007, 2): "9b3a98bc018d6cf2b11d76844780dfdd",
        (2023, 2007, 3): "270a53e6881b228074991d168e987a7d",
        (2023, 2007, 4): "6edb46b27bde00d1fbfb42ace37e7e3c",
        (2023, 2007, 5): "b953a337ce351c991683700f27e50900",
        (2023, 2007, 6): "588c8056e2dae9aaf303b84675808278",
        (2023, 2007, 7): "89096c81e431cf00303ea1e7b27e5b60",
        (2023, 2007, 8): "4d44aa1b8169fecb83071cb9d6f926c6",
        (2023, 2007, 9): "ca9b678b07c631b74abd6d7c125c7506",
        (2023, 2007, 10): "aa0312a4f3712603969f94427ba89848",
        (2023, 2007, 11): "862a7ba693a0f237a107f9614af55f5b",
        (2023, 2007, 12): "35e4fe88f83fe752bc8cd8e42b745c95",
        # Year 2006
        (2023, 2006, 1): "4f6f10861acf79131c0a8366a7883015",
        (2023, 2006, 2): "06206404983c0776856c6b576cf16696",
        (2023, 2006, 3): "2d31eb375ec649d439236263a15d6765",
        (2023, 2006, 4): "91f9e4421e9cc743c0640547edc79e9c",
        (2023, 2006, 5): "d49f3eedcc8624f35a9916e482c416be",
        (2023, 2006, 6): "df762309f73337b64733dcb6259f1c43",
        (2023, 2006, 7): "eafdc3b1b72789b069505ba8d40f7304",
        (2023, 2006, 8): "f54f3e073217efdea4d875a5ccf1ec94",
        (2023, 2006, 9): "4692cba61c2b20960a333b6f9a3387cf",
        (2023, 2006, 10): "c48536ce2087afb056d11b81552737b0",
        (2023, 2006, 11): "143f7692026d2906724bb7752f9abd66",
        (2023, 2006, 12): "b3a1615b9cb8275559b01c8a5836847a",
        # Year 2005
        (2023, 2005, 1): "21c607ec166f42d7188063ee195c841f",
        (2023, 2005, 2): "e7c94971d3d16b9a39707813e09854da",
        (2023, 2005, 3): "d9598a3eb6db5a854252c1f2ddd55ffe",
        (2023, 2005, 4): "901d86e21bf1e97faf070ba829b704c1",
        (2023, 2005, 5): "c470c79ab5e357cc8def14da9a2cdb40",
        (2023, 2005, 6): "a943d55bb7ce131638aabeeaafae822e",
        (2023, 2005, 7): "806f904cfe0f7bc301741a907ba72044",
        (2023, 2005, 8): "b21e293753802523a18da3950d93903f",
        (2023, 2005, 9): "555b2daaf0a6f9bd1a655533a735b84b",
        (2023, 2005, 10): "1311f0fff667b7fda4f639520cac6c84",
        (2023, 2005, 11): "dda2ecaf962dc12c906d153fa0f7061b",
        (2023, 2005, 12): "773c59a8d83b478f75a32cc12e30c7f6",
        # Year 2004
        (2023, 2004, 1): "0dd30909947aa2c25eb05a07e1845e23",
        (2023, 2004, 2): "7f806b37385bb9cabc09affba4d753d6",
        (2023, 2004, 3): "d708af3d64cf4e22b573d58ab4da540b",
        (2023, 2004, 4): "1421e0328e81221cf8975076a1f712ae",
        (2023, 2004, 5): "c2b90b3f824ddfcb0a9aa1364b9fbb77",
        (2023, 2004, 6): "8e74d782891de753aa61465664378a0a",
        (2023, 2004, 7): "ad58510012e2e553749c4a52b4f7b94a",
        (2023, 2004, 8): "bbfd68d41da60cf51d9237313ca6a25d",
        (2023, 2004, 9): "ae07d6a70489d5d724d802953daaf2b7",
        (2023, 2004, 10): "8f2172b8bf902fe59db5f90ef7c39776",
        (2023, 2004, 11): "0ee834184e5e792df29546a263a78ff0",
        (2023, 2004, 12): "9f868e2d4116eee4b2d4cd54a8edc376",
        # Year 2003
        (2023, 2003, 1): "59347880ba85db5bab55a072bc5b69b0",
        (2023, 2003, 2): "2c904f43fb03ad2c63dd9f0df51646cc",
        (2023, 2003, 3): "e59dc75ba84bef9d3b197322e626ea19",
        (2023, 2003, 4): "f87e33d5922aa22f3eed90bf2b49be35",
        (2023, 2003, 5): "3a17ebb103e4f57dea8f839592c6cab7",
        (2023, 2003, 6): "44c6231ea7f1f5495a1913111374dc7e",
        (2023, 2003, 7): "3837a2bb6eaf59c4ec31ad010d645246",
        (2023, 2003, 8): "744ecb465dbc5ef34c0e4e1d038fc452",
        (2023, 2003, 9): "aec045cd593ed5c390c12287de4313a5",
        (2023, 2003, 10): "3076aced8f9f37955e0f85501c3ee785",
        (2023, 2003, 11): "14287186bd92da9213bbb186a1d0f65b",
        (2023, 2003, 12): "ca73c12fb42eaa9fd5c577a4c461399c",
        # Year 2002
        (2023, 2002, 1): "38b788c40c24a3cc524954079c8e611c",
        (2023, 2002, 2): "087fff57b3445a14e62b52479d828bbc",
        (2023, 2002, 3): "668e85817163a019b305cadb571ff41d",
        (2023, 2002, 4): "8e8f98a938c74f00651cbe7292ec98c6",
        (2023, 2002, 5): "e8a22bda1ac3279b4df85dcba02c22f8",
        (2023, 2002, 6): "b257a1b7508dc8fd34cb218073611afc",
        (2023, 2002, 7): "370a1e9b797034363a001a483ed473c4",
        (2023, 2002, 8): "2b58a53aec11267b4e8ea6e36b666a5d",
        (2023, 2002, 9): "a342aa27ebfbe6a4c0b264c2a44aef37",
        (2023, 2002, 10): "d621298e95154475e304f8bc31c45b60",
        (2023, 2002, 11): "7ab5bf903c6e25a4b9a79ca2eab87261",
        (2023, 2002, 12): "dde203e187c79d81abb70be909944dd7",
        # Year 2001
        (2023, 2001, 1): "634bc647fbc904080b3fe6a6dc8a8321",
        (2023, 2001, 2): "39f4c3988e0617760ad0f36824666680",
        (2023, 2001, 3): "c4ef9ef98e05349c9de71090627efa44",
        (2023, 2001, 4): "2d8171dfe7487822f6984b403da23829",
        (2023, 2001, 5): "219862f215137c9afea25001c056e525",
        (2023, 2001, 6): "fcadd49993e59428e5936570c9279d10",
        (2023, 2001, 7): "b1a958a6e9c4f43957292c82c36143ab",
        (2023, 2001, 8): "422e7ee7febda80e6efbb85060d531c2",
        (2023, 2001, 9): "5127ba4327329ae7e2d72f09d9e0c565",
        (2023, 2001, 10): "eec34d5ca451290a748825abc4455108",
        (2023, 2001, 11): "27b2af1a85be35643969b912f28d811f",
        (2023, 2001, 12): "7f84a4eaccdfbdade8c48ffe5c4222e0",
        # Year 2000
        (2023, 2000, 1): "6aaa315ee69d20b1bf4634d8c575cca3",
        (2023, 2000, 2): "6bccb3ff4ff7847f4d732d0ed9687ea5",
        (2023, 2000, 3): "f79ec6c1c9de3320f1d9a0b6b821c5ef",
        (2023, 2000, 4): "a0d50afd466763b77306dd5c059fe06e",
        (2023, 2000, 5): "7ec491788a39535126f73d9eece2b122",
        (2023, 2000, 6): "17d40afb7f6457e45e0cbc22ce9376ba",
        (2023, 2000, 7): "16aa95d30ddba70664b6f51fd9ae180b",
        (2023, 2000, 8): "0de268d1976b484bd9bac5736ae8c338",
        (2023, 2000, 9): "644e2494537d479486fe37ff6d506afc",
        (2023, 2000, 10): "5817f0a7a5355265747ffbc7b0b1eceb",
        (2023, 2000, 11): "83ebb9344050949b82c1d3da6911beaa",
        (2023, 2000, 12): "cbd4dca0dbe7e5271afd534f01b3c337",

        # --- ODIAC2022 ---
        # Year 2021
        (2022, 2021, 1): "84afc4f461d40017a496ba5433a8c81b",
        (2022, 2021, 2): "46e174494324f4e764cf4921a675f837",
        (2022, 2021, 3): "ff3c9345216059853143fb36c510609c",
        (2022, 2021, 4): "d9a4ee2c37974a51ce24c9af3f1c9780",
        (2022, 2021, 5): "f984b08c1d38db405b36a530da0fc127",
        (2022, 2021, 6): "9159039335955f19a29a1f5e1ca36675",
        (2022, 2021, 7): "eb77eb7ddc6b80aa064aeb61dc2d81eb",
        (2022, 2021, 8): "04e4d24d607059d1a25b9f907aa56b30",
        (2022, 2021, 9): "a14e95c4b5d8a8a769b446265755d1f1",
        (2022, 2021, 10): "7263dda96243ecea811c4afd53826a75",
        (2022, 2021, 11): "c023f3ce31dfacd57da2aed35cf9b778",
        (2022, 2021, 12): "77b14f18605455a8fb4a62a7b66708f5",
        # Year 2020
        (2022, 2020, 1): "9dc9fb077a97a13a9fadecfac8cd6a2e",
        (2022, 2020, 2): "d18358cb36a27fb1dab9975a260a3a06",
        (2022, 2020, 3): "6e6afc01cec12960885b4ec5d5988dc4",
        (2022, 2020, 4): "211ae6c02f62b5e1d89b2e38fd9c7f89",
        (2022, 2020, 5): "84e05a579940c78c6efb8755e05f6438",
        (2022, 2020, 6): "0864149b099466febafd9e3b45fd32f5",
        (2022, 2020, 7): "2b0e9bf0dc620abaae575aff115f125c",
        (2022, 2020, 8): "cf34b512461eec1796bb96c753e56591",
        (2022, 2020, 9): "d1d5d7023c098ec12a3732d0f6eef585",
        (2022, 2020, 10): "cae2e5d62e61cde1d62b4744fa9c3ebc",
        (2022, 2020, 11): "083e9d2e8d6377e23b959bbd8a6269f4",
        (2022, 2020, 12): "1b2798b5a9ae12d395b599a067d2c41d",
        # Year 2019
        (2022, 2019, 1): "fe751cd2840c579bae1cb2af9dca37a6",
        (2022, 2019, 2): "775d70c96300f6ecee5191bb01df97dd",
        (2022, 2019, 3): "9a5fdd563da3187c485934956cc20f7f",
        (2022, 2019, 4): "40b8f02c83eff34a56e0ea3e854d4c48",
        (2022, 2019, 5): "d5c041601ef1ca9b052a376712e7f9ee",
        (2022, 2019, 6): "aed37a63f9eaec69f76a9bfb4e28ca80",
        (2022, 2019, 7): "dc19b811a3498bc43b8fa982fef62ac2",
        (2022, 2019, 8): "ade6cca7d55942e232f074701534094e",
        (2022, 2019, 9): "31e04e62a76bcaf6a0b666c1174c9ad8",
        (2022, 2019, 10): "8fbac244883febf2a0b255a594579372",
        (2022, 2019, 11): "ee271c719cc75d08ba26198d14855a52",
        (2022, 2019, 12): "40e6b9a068a95d38f95f6c3c21eba3d1",
        # Year 2018
        (2022, 2018, 1): "43369a04281af42785ae524c0e8c8bff",
        (2022, 2018, 2): "0f7c810d7e91a3136cedb9efc435a9e0",
        (2022, 2018, 3): "a84206fbd4330fedee342e2cd781406a",
        (2022, 2018, 4): "62c8f20a054b758f000ff9112df925a9",
        (2022, 2018, 5): "451c7f1bff55d1ee5c3b87ce5499276f",
        (2022, 2018, 6): "64bb327873bb30027e062105bcf53bf9",
        (2022, 2018, 7): "665f7507fec072c75751c12f4c7c0aaf",
        (2022, 2018, 8): "dd0d95d0cab19482b34efe579b4686e4",
        (2022, 2018, 9): "f156fd609ed0ee6e4f4afd3c008d3836",
        (2022, 2018, 10): "bb42133f96b19d72b46dcdb24b99b567",
        (2022, 2018, 11): "8f33c1d9f8414c5dac448789d930fd3a",
        (2022, 2018, 12): "236223d0399667617e4baa9f9c3c48cf",
        # Year 2017
        (2022, 2017, 1): "bb451d4dec256d2d393366d69b646b99",
        (2022, 2017, 2): "266d745910a3f3a2c9bb5f075ac54bc4",
        (2022, 2017, 3): "d4c10311711d91e929a70c3d47b19d2d",
        (2022, 2017, 4): "ef1f604c1842ef477342cf10fd6884be",
        (2022, 2017, 5): "e04f95cea5608105c4ead89eb2fedecb",
        (2022, 2017, 6): "8cfac2a881d1f75c4cd8977cc568516a",
        (2022, 2017, 7): "16e9e4763193d62fbc53a555b931db68",
        (2022, 2017, 8): "326152b686d0698ef2d5ab87c48cb76d",
        (2022, 2017, 9): "054b63878e0527a1934e921d9cc1c050",
        (2022, 2017, 10): "fa6197925ea19d2a6d72778ff882c8c4",
        (2022, 2017, 11): "1003f0271dcb8fd007f8bb4793f0bc84",
        (2022, 2017, 12): "07574c3cfbd5448fe64ad525eaf345b8",
        # Year 2016
        (2022, 2016, 1): "7796f3ef156721384dc2ed476a25fa79",
        (2022, 2016, 2): "039e2c7fe17df2f7beab23baaca4ef54",
        (2022, 2016, 3): "c0381ae3061d2d7296282ec758a4e80c",
        (2022, 2016, 4): "3385f561f8f3dd2b0ffaffafdc955531",
        (2022, 2016, 5): "4f2f978c57749b244250474a1c8b99e1",
        (2022, 2016, 6): "633b04a9e215b3793a640469a428a644",
        (2022, 2016, 7): "f07ae275b66614b13cca79741f621fd6",
        (2022, 2016, 8): "f994a14765466a8ddc3bfeb5b555fd4b",
        (2022, 2016, 9): "cbc359a60bd3eb0680df0af39afa9127",
        (2022, 2016, 10): "7b4ed27390fed89870dc4c6e843c0a38",
        (2022, 2016, 11): "3eea8bb885c54b189d040daeabac2321",
        (2022, 2016, 12): "7f090beb423690b30fe6de85d0a6d392",
        # Year 2015
        (2022, 2015, 1): "73dbafea75ba017353304872b6a39720",
        (2022, 2015, 2): "b98dc4ae2821fbe5cdaceee98d037760",
        (2022, 2015, 3): "be334630d2625ab380571063cdea61a7",
        (2022, 2015, 4): "682df1f5f632123e4582e2df223c3d4a",
        (2022, 2015, 5): "70e387de2918216d2f7e1439ef275847",
        (2022, 2015, 6): "2389a939e83c9deed842421507a8ff98",
        (2022, 2015, 7): "6524aaaca2eb7c2603629f9f7bfd6332",
        (2022, 2015, 8): "55de5c1e7f4e72d4c7924f3014ecb7d6",
        (2022, 2015, 9): "50a1b17af53750a9c80b563d34ce013f",
        (2022, 2015, 10): "42164e6ae8f9c6fdc415022167219fca",
        (2022, 2015, 11): "c92bd7af6a0f008163a1e36abaf51783",
        (2022, 2015, 12): "88d2684340ae05b7743f9d7d4ef9b597",
        # Year 2014
        (2022, 2014, 1): "dfb9db29f66f259599908f51afffc038",
        (2022, 2014, 2): "ca7dcc95ff4784dfd427d6775125dfe4",
        (2022, 2014, 3): "b9350c72052ddd98027a54022a7efc5f",
        (2022, 2014, 4): "a31573f920fe9d83c46d4151525904d3",
        (2022, 2014, 5): "8e28bb82e65bac24cde283959deb15bc",
        (2022, 2014, 6): "739df649f3513e4a50b1d3c8a877266e",
        (2022, 2014, 7): "b7633c41db868683e4ae1b2b385b43ed",
        (2022, 2014, 8): "c6e5a401e27c41c3c89b6f61bb4d0e34",
        (2022, 2014, 9): "b493f4e9bcd93cf682a13fbb1671bb78",
        (2022, 2014, 10): "935c3d993b845e20ab3a70c4d7d3c4ba",
        (2022, 2014, 11): "b2041c72cedb068d1f7c37601793ef42",
        (2022, 2014, 12): "079f2a79a90ad4dadf6f6e0d8dc84349",
        # Year 2013
        (2022, 2013, 1): "1d96755c5b8d56370c816dfdf23ce25e",
        (2022, 2013, 2): "ac62693264940c99f6fe116d051f1139",
        (2022, 2013, 3): "77c3a9c54ac6b246d47db960eef28803",
        (2022, 2013, 4): "513328d8ebf06752112731783ddef96e",
        (2022, 2013, 5): "6087a2c92558cec6adcfb4203912c124",
        (2022, 2013, 6): "bcce5856eed6cf0c6a830aa5c4178ffa",
        (2022, 2013, 7): "7ec8b5130809be7e6095f5da772978d8",
        (2022, 2013, 8): "9f9f31f6f719fd3f8181c8fa83917ec4",
        (2022, 2013, 9): "76beec549420a9a63ca49dcaa977cd46",
        (2022, 2013, 10): "4598d95e408d7e14fa33f3ebdf1bc46c",
        (2022, 2013, 11): "1dc400f3361ce6a245403d168008c4be",
        (2022, 2013, 12): "699073e09d4b6e12c0b402abb55645ad",
        # Year 2012
        (2022, 2012, 1): "ce773a660207fffe9ecbb807cd001872",
        (2022, 2012, 2): "59114376db9b1f35de8c309d8fa1bf9d",
        (2022, 2012, 3): "a0de9045630a9a8842373bbe5db27c81",
        (2022, 2012, 4): "49f119b69eecbaf1fb873a340c1fa47f",
        (2022, 2012, 5): "e273bf2643b8954048daf73f8c1b0efb",
        (2022, 2012, 6): "3f17f03a544172d17c2b59623652e5d2",
        (2022, 2012, 7): "03488f019ac07798a596d35babd40ad5",
        (2022, 2012, 8): "50aa1c1cee74caff226b54e667b9544b",
        (2022, 2012, 9): "b91d63f20ab91d098d004b038a98c0eb",
        (2022, 2012, 10): "ec77c369204c586370b8ec8f6897d0a5",
        (2022, 2012, 11): "2bad082224404b23cd9b26d0027afc2e",
        (2022, 2012, 12): "029dd1782165f7fcfc55dbbec109b9d2",
        # Year 2011
        (2022, 2011, 1): "3172b2273b63a05cd9b831a07dd7ef06",
        (2022, 2011, 2): "b0af509bb516fa9bebca53baf4cd241c",
        (2022, 2011, 3): "1edcebe0365254bd38cb71fd7b1c5783",
        (2022, 2011, 4): "8a2765de729f45659b471b7e6d8ce863",
        (2022, 2011, 5): "c985bc39ee4eb7d69c6a28bf8f38200e",
        (2022, 2011, 6): "54ba123f1a515bbd45d7251a82fd7b0d",
        (2022, 2011, 7): "db02f363a81eae580ed558b09d1417d5",
        (2022, 2011, 8): "cba18964b3dde6699bb75a99fc4cf968",
        (2022, 2011, 9): "f74e01f15bea463e262d97d295fcf519",
        (2022, 2011, 10): "0c5c878c90cbd10b6312674f3bfda83f",
        (2022, 2011, 11): "5df2651bfe109452d325aed176161cad",
        (2022, 2011, 12): "325942c8c5f3971680c0892c4c199269",
        # Year 2010
        (2022, 2010, 1): "ce7115b529d185ce79a2b22ffb16aecd",
        (2022, 2010, 2): "9d1071c257e1a0a2de289acb8f5b81e1",
        (2022, 2010, 3): "5b344e562603f48b0390c61809d6ed70",
        (2022, 2010, 4): "343efb9e840e8a8a6222571468e0cecf",
        (2022, 2010, 5): "3cb90705e507ffa046da034dfaf209a6",
        (2022, 2010, 6): "ed1082bfbc53fede8e3c719ea0842a5b",
        (2022, 2010, 7): "3e9e8f5d6ac06537328d4d614006ed22",
        (2022, 2010, 8): "8c19808f70003748caa0eabe85a4c3f0",
        (2022, 2010, 9): "b29007db673b3589151d39b2aec5ae43",
        (2022, 2010, 10): "dc9f53b266292ea4dc991093c6c5f864",
        (2022, 2010, 11): "7f9a137d0e87470383250b00759157c0",
        (2022, 2010, 12): "2509bd5c53ddd69e58e7093b393646d9",
        # Year 2009
        (2022, 2009, 1): "b5e15ec34a872c608b1a75a78ad90845",
        (2022, 2009, 2): "d4ec5058937fd6b843a4a509b3586ace",
        (2022, 2009, 3): "cfe426196da5d503ecd91d567d7ae7bd",
        (2022, 2009, 4): "ab63620c15d09f409ceeaea7db980db5",
        (2022, 2009, 5): "7d78cc72e01bef1546187a5fbc277864",
        (2022, 2009, 6): "a60ec8cb011f69b5416c1bb0127778d6",
        (2022, 2009, 7): "30e2fbf92d9d04132f07c8db3d0472ae",
        (2022, 2009, 8): "4e3cdccaf82e2c87f6813dfdea3d17df",
        (2022, 2009, 9): "baebc300c9a1f0617bbcde39d59b331d",
        (2022, 2009, 10): "8d037702c9b693f88586f45ea0b070f5",
        (2022, 2009, 11): "f7dc6b5d97889eefcce5a371d41db640",
        (2022, 2009, 12): "b6cdbc8cf4d2bcc3a3202b807e2933e1",
        # Year 2008
        (2022, 2008, 1): "0d5f6dd6443ba17564f9226d551f72a7",
        (2022, 2008, 2): "c7aa7365bbfc55e5b350b994953700c4",
        (2022, 2008, 3): "d71e8184fb0f41de9bf0a40858a99aaa",
        (2022, 2008, 4): "46d20ee5bdaf3c8c4805d67f8a7d0564",
        (2022, 2008, 5): "562a27089b8a2e13255a6cff42d165ab",
        (2022, 2008, 6): "0e5ad303e7ceb12dc5e562be3c2b90a4",
        (2022, 2008, 7): "902eded4f5ae1f388d322d60f433619b",
        (2022, 2008, 8): "6b8513d976309b941e7769f2ba1da84c",
        (2022, 2008, 9): "62f7d14d46f201e58d8b48f6b0febe0b",
        (2022, 2008, 10): "45911845febb73616726804d7f6d679b",
        (2022, 2008, 11): "edb5f36b574d48e271a8a171ca7fe223",
        (2022, 2008, 12): "fc03a695e897b535fdfdbcef1791126d",
        # Year 2007
        (2022, 2007, 1): "5566c91298346b2c1a03d0fe33c2e745",
        (2022, 2007, 2): "efe01e3562ba05954787ce10fb3c81e4",
        (2022, 2007, 3): "59f1b83e550bf41a65836dc3ccc4da2c",
        (2022, 2007, 4): "8524d65ff33e5a3177d7b57b5a9359bc",
        (2022, 2007, 5): "44bd3bc38955e0be5130b0d17784d85e",
        (2022, 2007, 6): "6a660bbe90be6f0e79fded324fd0bf10",
        (2022, 2007, 7): "32c96d3b96a50cecbcb88a68548cbb88",
        (2022, 2007, 8): "b28773923b1f9a816c7e06bd2b8fccea",
        (2022, 2007, 9): "6197e84bf4e1eb774334d1ea12893d28",
        (2022, 2007, 10): "1574308b2a8ccfbec1e2eb9d3dad0b9f",
        (2022, 2007, 11): "70e8d31eea6a688fcb4bc238b82aa84e",
        (2022, 2007, 12): "194d356b125fbc59cc49d9ccf275d0a2",
        # Year 2006
        (2022, 2006, 1): "29e3c798dedab968e8dee1b4e209c4b0",
        (2022, 2006, 2): "5d5885f22b76915411d97b356e15bba6",
        (2022, 2006, 3): "0d4f4dffd97ccf3452cf7e9d2ca38f46",
        (2022, 2006, 4): "f354d58b1f2d6d63a703a4122555f31d",
        (2022, 2006, 5): "476bab60ecbd178c9edf29c453a2b9c8",
        (2022, 2006, 6): "b3aa2d7acde65ebf0481c68aa226ac47",
        (2022, 2006, 7): "a017c04ac75609f9029724ff1fbbbf66",
        (2022, 2006, 8): "cacb803c3ffe1a4108a45fea85f1eb2a",
        (2022, 2006, 9): "bc007effb3affb0e3c9756cc92944c01",
        (2022, 2006, 10): "96f251e3f93a32ec276d49ab6bd34f6d",
        (2022, 2006, 11): "9fcb33b10a5760e1714411869fba0a2c",
        (2022, 2006, 12): "497c1e4a67420b3ceb4645021ab1beab",
        # Year 2005
        (2022, 2005, 1): "d77920a2362e0c24dc228063eba29190",
        (2022, 2005, 2): "a9734e7983109dfcfa20d29322c34d92",
        (2022, 2005, 3): "676f932d5446025a05ef1c7f918c7add",
        (2022, 2005, 4): "669877accdf8fb0ff6ce037b785ce9e9",
        (2022, 2005, 5): "5bc31b5c053f310908a048cab70283e3",
        (2022, 2005, 6): "1899d3708f7f88aee938b1dd3d922af5",
        (2022, 2005, 7): "c1f247133830d5184b40fb096d2d5a4f",
        (2022, 2005, 8): "35bf5801f1e1222532456d016ca54c10",
        (2022, 2005, 9): "4f6762b46c69fd8078eb9706a5da7064",
        (2022, 2005, 10): "c41bfa24f7dcc2013ef6ed08e208f51d",
        (2022, 2005, 11): "38b6cf38f3933404484453709b9bff58",
        (2022, 2005, 12): "93f7419f218ca740cc686ba53e07b4ec",
        # Year 2004
        (2022, 2004, 1): "679fca2ddb42c2d4b1dc0ad6912cb017",
        (2022, 2004, 2): "42a9a26c429e1af6de80b94ed4e43033",
        (2022, 2004, 3): "b078d58a1724e4634f70f2427c8f88e4",
        (2022, 2004, 4): "910d74b7e05dd12ec00fd2bd4c5aad5a",
        (2022, 2004, 5): "fde0463cee06467dc8ca9a3e594083dd",
        (2022, 2004, 6): "6a2db3b31c59b653cc4446c66b78934e",
        (2022, 2004, 7): "8f6efe6eb9ee9a08487362169ef26874",
        (2022, 2004, 8): "ee00856b5263a3515886d572e4b35de6",
        (2022, 2004, 9): "1cea9a970ec4403651cae7c964946fb0",
        (2022, 2004, 10): "de3ea7b12b5eefea36716a1db8d40fa6",
        (2022, 2004, 11): "5a9c553abdadc6150e514a9273627e7d",
        (2022, 2004, 12): "50cef59076bbe8a964f9e88a094a08d1",
        # Year 2003
        (2022, 2003, 1): "569826224f785443518c1351cb2ca944",
        (2022, 2003, 2): "aea5ab8886fe561116110a7ceb89fb32",
        (2022, 2003, 3): "75e5f5843593d0e4a509a4d18604eb2e",
        (2022, 2003, 4): "184ba6f449d4f7b5fa3f4c86be36e2bc",
        (2022, 2003, 5): "e4168db960e702197c67cfa5fdfd0cfb",
        (2022, 2003, 6): "c6fe4e16f03d94ab4f7d8da5ebc6c80b",
        (2022, 2003, 7): "0b44b13d77b79c596fa9faccbe18bf87",
        (2022, 2003, 8): "249a202ae0c9cb67a0d24660281749ea",
        (2022, 2003, 9): "c7625808ba00336db4b7b79b771292eb",
        (2022, 2003, 10): "01d1766cd88efa325263d6f7ffb58c45",
        (2022, 2003, 11): "fe3f5b7abf029f9024a9ef120b16133f",
        (2022, 2003, 12): "8bc71fa184f85243be8d428a18ea2eee",
        # Year 2002
        (2022, 2002, 1): "50b700cc6a97a81b904c5a3f656f1530",
        (2022, 2002, 2): "ca67ef25fdc1d30cbbe4762de2edbe38",
        (2022, 2002, 3): "71520bd5ee5ead5100dadf2b982cba48",
        (2022, 2002, 4): "39c80269a50914385a0b44d79351db20",
        (2022, 2002, 5): "4f495f1705f14e08dedcb06504b18e0f",
        (2022, 2002, 6): "254a30316373a34ea7cb75bedaeada6e",
        (2022, 2002, 7): "4dbfd5cf07403b9110211bd5f36307b2",
        (2022, 2002, 8): "a2bb6d3fda3a1b930e04e9cf01e222c2",
        (2022, 2002, 9): "52ad8cf6459f15569fdd3ae8f062886e",
        (2022, 2002, 10): "07c5174bf39278b7ddfc4cb7484df9d8",
        (2022, 2002, 11): "5bcd429df5218cb0e94f770575fc083c",
        (2022, 2002, 12): "1cdec12f084d17639a90c8a02c698753",
        # Year 2001
        (2022, 2001, 1): "05fdac171884d34a5c3de1bad9292a9c",
        (2022, 2001, 2): "85d1552d48fd6fd1d62f46b2ad76345a",
        (2022, 2001, 3): "cb625f489ff9c36f33dd6fd42b49ed7e",
        (2022, 2001, 4): "46a55e86131ec27dc9bbc83d6ad59ca5",
        (2022, 2001, 5): "de26c3c0c8f77ee5f6fc9396dfd759fb",
        (2022, 2001, 6): "512387d22c14603b50b6bff089ca291d",
        (2022, 2001, 7): "dcb44169433df63e769ad236c9b7825e",
        (2022, 2001, 8): "9314f14d4434c9725c8a6f527ff9a711",
        (2022, 2001, 9): "cdbea55de763225f03844d70a632138a",
        (2022, 2001, 10): "5d9b88971e0067376ca3b168709f00ca",
        (2022, 2001, 11): "103dab5897c12c5956b58c72e1db8020",
        (2022, 2001, 12): "d19d3c54ea7c1857d85aa43e5fed5b60",
        # Year 2000
        (2022, 2000, 1): "0c8258465b69ac4ba015956007b49e63",
        (2022, 2000, 2): "c9ca9211c409071ea758a68c5f69ca92",
        (2022, 2000, 3): "05f1866c6b4b7cab611ba508fbb99055",
        (2022, 2000, 4): "049125744ee4f38326c3105a7828a9a5",
        (2022, 2000, 5): "b5c3239b0368e4377dbd2f7017adff7d",
        (2022, 2000, 6): "051a58924cce04801cd99ee6904dfb40",
        (2022, 2000, 7): "475038a1495096f2b494290b9ee493e6",
        (2022, 2000, 8): "da345a07ff58b86475a260be3e43d201",
        (2022, 2000, 9): "512df253c2d5e4902f6733f801d082d6",
        (2022, 2000, 10): "caebc46d8e7726ec686d514324d054ab",
        (2022, 2000, 11): "37316bd62231f61842a68fe2f85f7b59",
        (2022, 2000, 12): "c0df1664ddc80d57b4db1c641429b33d",
    }

    # Supported versions and the range of years available for each
    valid_versions: ClassVar[dict[int, tuple[int, int]]] = {
        2023: (2000, 2022),
        2022: (2000, 2021),
        # Add older versions if desired
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        version: int = 2023,
        years: Sequence[int] | None = None,
        months: Sequence[int] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: Root directory wherein dataset can be found or downloaded. Subfolders
                   for each year (e.g. '2021', '2022') are expected within this root.
            crs: Coordinate reference system (CRS) to warp to. Defaults to
                 EPSG:4326 (WGS84).
            res: Resolution of the dataset in units of CRS. Defaults to
                 ``0.008333333333333`` (approx 1km).
            version: ODIAC dataset version year (e.g., 2023). Determines the latest
                     available year (version - 1).
            years: Year(s) to include, ranging from 2000 to (version - 1).
                   Defaults to all available years for the selected version.
            months: Month(s) to include (1-12). Defaults to all months.
            transforms: A function/transform that takes input sample and its target as
                        entry and returns a transformed version.
            cache: If True, cache file handle to speed up repeated sampling.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            AssertionError: If 'version', 'years' or 'months' are invalid.
            DatasetNotFoundError: If dataset is not found and download is False.
        """
        assert version in self.valid_versions, (
            f"Invalid version {version}. Choose from {list(self.valid_versions.keys())}"
        )
        min_year, max_year = self.valid_versions[version]
        if years is None:
            years = list(range(min_year, max_year + 1))
        else:
            for year in years:
                assert min_year <= year <= max_year, (
                  f"Invalid year {year} for version {version}. "
                  f"Valid years are between {min_year} and {max_year}."
                )
        self.years = years

        if months is None:
            months = list(range(1, 13))
        else:
            for month in months:
                assert 1 <= month <= 12, f"Invalid month {month}. Must be 1-12."
        self.months = months

        self.paths = paths
        self.version = version
        self.download = download
        self.checksum = checksum

        self._crs = crs or CRS.from_epsg(4326)
       # Set default resolution *based on the target CRS* if not provided
        # If the target CRS is geographic (like 4326), use degrees default
        # Otherwise, assume a projected CRS and use a meter-based default (e.g., 1000m)
        if res is None:
            if self._crs.is_geographic:
                self._res = (0.008333333333333, 0.008333333333333)
            else:
                # Default for projected CRS - adjust if ODIAC data is ever
                # commonly used in a specific non-geographic projection.
                self._res = (1000.0, 1000.0)
        elif isinstance(res, (int, float)):
            self._res = (float(res), float(res))
        else:
            self._res = res

        self._verify() # Check for data, download/extract if needed

        # Initialize GeoDataset parent class AFTER verifying/downloading
        # This sets up transforms and index R-tree.
        GeoDataset.__init__(self, transforms=transforms)

        # We manually build the index AFTER parent init and verification
        self._build_index()

        # Set RasterDataset specific attributes (needed if using methods from it)
        self.cache = cache
        self.bands = self.all_bands
        self.band_indices = [1] # Always just the first band

    def _build_index(self) -> None:
        """Build the R-tree index for the dataset by manually scanning files."""
        assert isinstance(self.paths, (str, os.PathLike)), "'paths' must be a single path for ODIAC dataset indexing."
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        idx = 0
        root_path = PathlibPath(self.paths)

        for year in self.years:
            year_dir = root_path / str(year)
            if not year_dir.is_dir():
                # Don't warn if download=False, as user might provide specific file paths
                if self.download:
                     warnings.warn(f"Directory for year {year} not found, skipping index build for this year.")
                continue

            for month in self.months:
                month_str = f'{month:02d}'
                # Construct expected filename for this month/year/version
                # Note: YYMM format is required by the filename regex
                yymm_str = f"{str(year)[-2:]}{month_str}"
                expected_filename = f"odiac{self.version}_1km_excl_intl_{yymm_str}.tif"
                filepath = year_dir / expected_filename

                if not filepath.exists():
                     # Don't warn if download=False
                     if self.download:
                          warnings.warn(f"Expected file not found: {filepath}")
                     continue

                match = filename_regex.match(filepath.name)
                if match:
                    # Double check version match if needed
                    file_version = int(match.group('version'))
                    if file_version != self.version:
                         warnings.warn(f"File version mismatch: expected {self.version}, found {file_version} in {filepath.name}")
                         continue

                    try:
                        with rasterio.open(filepath) as src:
                            # Use the dataset's target CRS, transform bounds if needed
                            src_crs = CRS.from_dict(src.crs)
                            minx, miny, maxx, maxy = src.bounds

                            if src_crs != self.crs:
                                transformer = Transformer.from_crs(src_crs, self.crs, always_xy=True)
                                minx, miny, maxx, maxy = transformer.transform_bounds(minx, miny, maxx, maxy)

                            # Use the known full year and parsed month for timestamps
                            date_str = f"{year}{month_str}"
                            mint, maxt = disambiguate_timestamp(date_str, '%Y%m')

                            coords = (minx, maxx, miny, maxy, mint, maxt)
                            self.index.insert(idx, coords, str(filepath))
                            idx += 1
                    except rasterio.RasterioIOError:
                        warnings.warn(f"Could not open file {filepath}, skipping.")
                    except Exception as e:
                        warnings.warn(f"Error processing file {filepath}: {e}, skipping.")

        if idx == 0:
            # Raise only if download was attempted or if the root path exists but is empty
            if self.download or (root_path.exists() and not any(root_path.iterdir())):
                raise DatasetNotFoundError(self)
            else: # If a specific non-existent path was given without download=True
                warnings.warn("No matching files found for the specified paths, years, months, and version.")

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        assert isinstance(self.paths, (str, os.PathLike)), "'paths' must be a single path for ODIAC dataset verification."
        root_path = PathlibPath(self.paths)
        needs_download = False
        needs_extraction = False

        # Check for final extracted .tif files
        tif_files_present = {}
        for year in self.years:
            year_dir = root_path / str(year)
            tif_files_present[year] = False
            if year_dir.is_dir() and any(year_dir.glob(self.filename_glob)):
             tif_files_present[year] = True

        if all(tif_files_present.get(y, False) for y in self.years):
            print("Found extracted .tif files.")
            return

        # Check for compressed .tif.gz files
        gz_files_present = {}
        for year in self.years:
            year_dir = root_path / str(year)
            gz_files_present[year] = False
            if year_dir.is_dir():
                for month in self.months:
                    yymm_str = f"{str(year)[-2:]}{month:02d}"
                    fname = f"odiac{self.version}_1km_excl_intl_{yymm_str}.tif.gz"
                    if (year_dir / fname).exists():
                        gz_files_present[year] = True
                        needs_extraction = True
                        break # Found at least one .gz for this year

        if needs_extraction and all(gz_files_present.get(y, False) or tif_files_present.get(y, False) for y in self.years):
            print("Found compressed .tif.gz files, attempting extraction.")
            self._extract()
            # Re-check for tif files after extraction attempt
            tif_files_present_after_extract = {}
            for year in self.years:
                 year_dir = root_path / str(year)
                 tif_files_present_after_extract[year] = False
                 if year_dir.is_dir():
                      for month in self.months:
                           yymm_str = f"{str(year)[-2:]}{month:02d}"
                           fname = f"odiac{self.version}_1km_excl_intl_{yymm_str}.tif"
                           if (year_dir / fname).exists():
                                tif_files_present_after_extract[year] = True
                                break
            if all(tif_files_present_after_extract.values()):
                 return # Extraction successful for all needed years

        # If files are still missing, check if download is allowed
        if not self.download:
             # Raise error only if tif files are missing for ALL requested years
             if not any(tif_files_present.values()):
                  raise DatasetNotFoundError(self)
             else:
                  # Warn if some years are missing but download=False
                  missing_years = [y for y in self.years if not tif_files_present.get(y, False)]
                  if missing_years:
                       warnings.warn(f"Dataset files missing for years: {missing_years}. Set download=True to attempt download.")
                  return # Proceed with available data


        # Download and extract
        print("Attempting to download missing data...")
        self._download()
        self._extract()

        # Final check after download and extract attempt
        final_check_ok = True
        for year in self.years:
             year_dir = root_path / str(year)
             year_ok = False
             if year_dir.is_dir():
                  for month in self.months:
                       yymm_str = f"{str(year)[-2:]}{month:02d}"
                       fname = f"odiac{self.version}_1km_excl_intl_{yymm_str}.tif"
                       if (year_dir / fname).exists():
                            year_ok = True
                            break
             if not year_ok:
                  final_check_ok = False
                  warnings.warn(f"Verification failed: No .tif files found for year {year} after download/extraction.")

        if not final_check_ok:
            raise DatasetNotFoundError(self)

    def _download(self) -> None:
        """Download the monthly dataset files."""
        assert isinstance(self.paths, (str, os.PathLike)), "'paths' must be a single path for ODIAC dataset download."
        root_path = PathlibPath(self.paths)
        for year in self.years:
            year_dir = root_path / str(year)
            os.makedirs(year_dir, exist_ok=True) # Ensure year directory exists

            # Check if we already have all necessary tif files for this year
            all_tif_exist_for_year = True
            for month in self.months:
                 yymm_str = f"{str(year)[-2:]}{month:02d}"
                 fname_tif = f"odiac{self.version}_1km_excl_intl_{yymm_str}.tif"
                 if not (year_dir / fname_tif).exists():
                      all_tif_exist_for_year = False
                      break
            if all_tif_exist_for_year:
                 print(f"All .tif files already present for year {year}, skipping download.")
                 continue # Skip download for this year

            # Download missing .tif.gz files for the year
            print(f"Checking/Downloading data for year {year}...")
            for month in self.months:
                yymm_str = f"{str(year)[-2:]}{month:02d}"
                filename_gz = f"odiac{self.version}_1km_excl_intl_{yymm_str}.tif.gz"
                filepath_gz = year_dir / filename_gz

                # Skip download if final .tif already exists
                filepath_tif = year_dir / f"odiac{self.version}_1km_excl_intl_{yymm_str}.tif"
                if filepath_tif.exists():
                    continue

                # Download if .tif.gz doesn't exist
                if not filepath_gz.exists():
                    key = (self.version, year, month)
                    if key not in self.md5s and self.checksum:
                        warnings.warn(
                           f"MD5 checksum not available for version {self.version}, year {year}, month {month}. "
                           "Skipping checksum verification for this file."
                        )
                        md5_hash = None
                    else:
                        md5_hash = self.md5s.get(key) if self.checksum else None

                    try:
                        file_url = self.url.format(version=self.version, year=year, yymm=yymm_str)
                        download_url(
                            file_url,
                            str(year_dir), # Download directly into year folder
                            filename=filename_gz,
                            md5=md5_hash,
                        )
                    except Exception as e:
                        print(f"Error downloading {filename_gz} for year {year}: {e}")
                        # Optionally remove partially downloaded file?
                        if filepath_gz.exists():
                             filepath_gz.unlink()


    def _extract(self) -> None:
        """Extract the monthly .tif.gz files."""
        assert isinstance(self.paths, (str, os.PathLike)), "'paths' must be a single path for ODIAC dataset extraction."
        root_path = PathlibPath(self.paths)
        print("Extracting downloaded .tif.gz files...")
        extracted_count = 0
        skipped_count = 0
        failed_count = 0

        for year in self.years:
            year_dir = root_path / str(year)
            if not year_dir.is_dir():
                continue

            for month in self.months:
                yymm_str = f"{str(year)[-2:]}{month:02d}"
                filename_base = f"odiac{self.version}_1km_excl_intl_{yymm_str}"
                gz_path = year_dir / f"{filename_base}.tif.gz"
                tif_path = year_dir / f"{filename_base}.tif"

                if tif_path.exists():
                    skipped_count += 1
                    continue # Skip if already extracted

                if gz_path.exists():
                    try:
                        with gzip.open(gz_path, 'rb') as f_in:
                            with open(tif_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        extracted_count += 1
                        # Optionally remove the .gz file after successful extraction
                        # gz_path.unlink()
                    except Exception as e:
                        warnings.warn(f"Failed to extract {gz_path}: {e}")
                        failed_count += 1
                        # Optionally remove corrupted output .tif file if extraction failed
                        if tif_path.exists():
                             tif_path.unlink(missing_ok=True)

        print(f"Extraction summary: Extracted={extracted_count}, Skipped={skipped_count}, Failed={failed_count}")


    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`RasterDataset.__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional suptitle to use for figure.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        # Get the single band (adjust key based on is_image)
        image = sample['image'].squeeze().numpy() # Squeeze the channel dim
        ncols = 1

        # Determine robust color limits, e.g., using percentiles > 0
        valid_pixels = image[~np.isnan(image) & (image > 0)] # Exclude NaNs and zeros
        if valid_pixels.size > 1: # Need at least 2 distinct values for percentile
             vmin = np.percentile(valid_pixels, 1)
             vmax = np.percentile(valid_pixels, 99)
             # Ensure vmin < vmax, handle near-constant cases
             if vmin >= vmax:
                  vmin = valid_pixels.min()
                  vmax = valid_pixels.max()
             # Ensure vmin/vmax aren't identical if possible
             if vmin == vmax and valid_pixels.size > 0:
                  vmin = max(0, vmin - 1e-6) # Ensure non-negative vmin
                  vmax += 1e-6
             # Ensure vmin is at least 0, as negative emissions aren't typical here
             vmin = max(0, vmin)
        elif valid_pixels.size == 1:
             vmin = max(0, valid_pixels[0] - 1e-6)
             vmax = valid_pixels[0] + 1e-6
        else: # No valid pixels > 0
             vmin, vmax = 0, 1 # Default if no valid pixels or all zero

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = sample['prediction'].squeeze().numpy()
            ncols = 2
            # Optionally adjust vmin/vmax based on prediction range too

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 4), squeeze=False)

        im = axs[0, 0].imshow(image, cmap='magma', vmin=vmin, vmax=vmax)
        axs[0, 0].axis('off')
        plt.colorbar(im, ax=axs[0, 0], label='CO2 Emission (tonnes C / yr / cell)')

        if show_titles:
            axs[0, 0].set_title('CO2 Emission')

        if showing_predictions:
            im_pred = axs[0, 1].imshow(pred, cmap='magma', vmin=vmin, vmax=vmax)
            axs[0, 1].axis('off')
            plt.colorbar(im_pred, ax=axs[0, 1], label='CO2 Emission (tonnes C / yr / cell)')
            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout() # Adjust layout after adding colorbars
        return fig