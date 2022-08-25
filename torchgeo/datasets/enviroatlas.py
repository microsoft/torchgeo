# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EnviroAtlas High-Resolution Land Cover datasets."""

import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.mask
import shapely.geometry
import shapely.ops
import torch
from matplotlib.colors import ListedColormap
from rasterio.crs import CRS

from .geo import GeoDataset
from .utils import BoundingBox, download_url, extract_archive


class EnviroAtlas(GeoDataset):
    """EnviroAtlas dataset covering four cities with prior and weak input data layers.

    The `EnviroAtlas
    <https://doi.org/10.5281/zenodo.5778192>`__ dataset contains NAIP aerial imagery,
    NLCD land cover labels, OpenStreetMap roads, water, waterways, and waterbodies,
    Microsoft building footprint labels, high-resolution land cover labels from the
    EPA EnviroAtlas dataset, and high-resolution land cover prior layers.

    This dataset was organized to accompany the 2022 paper, `"Resolving label
    uncertainty with implicit generative models"
    <https://openreview.net/forum?id=AEa_UepnMDX>`_. More details can be found at
    https://github.com/estherrolf/qr_for_landcover.

    If you use this dataset in your research, please cite the following paper:

    * https://openreview.net/forum?id=AEa_UepnMDX

    .. versionadded:: 0.3
    """

    url = "https://zenodo.org/record/5778193/files/enviroatlas_lotp.zip?download=1"
    filename = "enviroatlas_lotp.zip"
    md5 = "bfe601be21c7c001315fc6154be8ef14"

    crs = CRS.from_epsg(3857)
    res = 1

    valid_prior_layers = ["prior", "prior_no_osm_no_buildings"]

    valid_layers = [
        "naip",
        "nlcd",
        "roads",
        "water",
        "waterways",
        "waterbodies",
        "buildings",
        "lc",
    ] + valid_prior_layers

    cities = [
        "pittsburgh_pa-2010_1m",
        "durham_nc-2012_1m",
        "austin_tx-2012_1m",
        "phoenix_az-2010_1m",
    ]
    splits = (
        [f"{state}-train" for state in cities[:1]]
        + [f"{state}-val" for state in cities[:1]]
        + [f"{state}-test" for state in cities]
        + [f"{state}-val5" for state in cities]
    )

    # these are used to check the integrity of the dataset
    files = [
        "austin_tx-2012_1m-test_tiles-debuffered",
        "austin_tx-2012_1m-val5_tiles-debuffered",
        "durham_nc-2012_1m-test_tiles-debuffered",
        "durham_nc-2012_1m-val5_tiles-debuffered",
        "phoenix_az-2010_1m-test_tiles-debuffered",
        "phoenix_az-2010_1m-val5_tiles-debuffered",
        "pittsburgh_pa-2010_1m-test_tiles-debuffered",
        "pittsburgh_pa-2010_1m-train_tiles-debuffered",
        "pittsburgh_pa-2010_1m-val5_tiles-debuffered",
        "pittsburgh_pa-2010_1m-val_tiles-debuffered",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_a_naip.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_b_nlcd.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_c_roads.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_d1_waterways.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_d2_waterbodies.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_d_water.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_e_buildings.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_h_highres_labels.tif",
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_prior_from_cooccurrences_101_31.tif",  # noqa: E501
        "austin_tx-2012_1m-test_tiles-debuffered/3009726_sw_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif",  # noqa: E501
        "spatial_index.geojson",
    ]

    p_src_crs = pyproj.CRS("epsg:3857")
    p_transformers = {
        "epsg:26917": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26917"), always_xy=True
        ).transform,
        "epsg:26918": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26918"), always_xy=True
        ).transform,
        "epsg:26914": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26914"), always_xy=True
        ).transform,
        "epsg:26912": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26912"), always_xy=True
        ).transform,
    }

    # used to convert the 10 high-res classes labeled as [0, 10, 20, 30, 40, 52, 70, 80,
    # 82, 91, 92] to sequential labels [0, ..., 10]
    raw_enviroatlas_to_idx_map: "np.typing.NDArray[np.uint8]" = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            6,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            7,
            0,
            8,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            9,
            10,
        ],
        dtype=np.uint8,
    )

    highres_classes = [
        "Unclassified",
        "Water",
        "Impervious Surface",
        "Soil and Barren",
        "Trees and Forest",
        "Shrubs",
        "Grass and Herbaceous",
        "Agriculture",
        "Orchards",
        "Woody Wetlands",
        "Emergent Wetlands",
    ]
    highres_cmap = ListedColormap(
        [
            [1.00000000, 1.00000000, 1.00000000],
            [0.00000000, 0.77254902, 1.00000000],
            [0.61176471, 0.61176471, 0.61176471],
            [1.00000000, 0.66666667, 0.00000000],
            [0.14901961, 0.45098039, 0.00000000],
            [0.80000000, 0.72156863, 0.47450980],
            [0.63921569, 1.00000000, 0.45098039],
            [0.86274510, 0.85098039, 0.22352941],
            [0.67058824, 0.42352941, 0.15686275],
            [0.72156863, 0.85098039, 0.92156863],
            [0.42352941, 0.62352941, 0.72156863],
        ]
    )

    def __init__(
        self,
        root: str = "data",
        splits: Sequence[str] = ["pittsburgh_pa-2010_1m-train"],
        layers: Sequence[str] = ["naip", "prior"],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        prior_as_input: bool = False,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            splits: a list of strings in the format "{state}-{train,val,test}"
                indicating the subset of data to use, for example "ny-train"
            layers: a list containing a subset of ``valid_layers`` indicating which
                layers to load
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            prior_as_input: bool describing whether the prior is used as an input (True)
                or as supervision (False)
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
            AssertionError: if ``splits`` or ``layers`` are not valid
        """
        for split in splits:
            assert split in self.splits
        assert all([layer in self.valid_layers for layer in layers])
        self.root = root
        self.layers = layers
        self.cache = cache
        self.download = download
        self.checksum = checksum
        self.prior_as_input = prior_as_input

        self._verify()

        super().__init__(transforms)

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint: float = 0
        maxt: float = sys.maxsize
        with fiona.open(
            os.path.join(root, "enviroatlas_lotp", "spatial_index.geojson"), "r"
        ) as f:
            for i, row in enumerate(f):
                if row["properties"]["split"] in splits:
                    box = shapely.geometry.shape(row["geometry"])
                    minx, miny, maxx, maxy = box.bounds
                    coords = (minx, maxx, miny, maxy, mint, maxt)

                    self.index.insert(
                        i,
                        coords,
                        {
                            "naip": row["properties"]["naip"],
                            "nlcd": row["properties"]["nlcd"],
                            "roads": row["properties"]["roads"],
                            "water": row["properties"]["water"],
                            "waterways": row["properties"]["waterways"],
                            "waterbodies": row["properties"]["waterbodies"],
                            "buildings": row["properties"]["buildings"],
                            "lc": row["properties"]["lc"],
                            "prior_no_osm_no_buildings": row["properties"][
                                "naip"
                            ].replace(
                                "a_naip",
                                "prior_from_cooccurrences_101_31_no_osm_no_buildings",
                            ),
                            "prior": row["properties"]["naip"].replace(
                                "a_naip", "prior_from_cooccurrences_101_31"
                            ),
                        },
                    )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        sample = {"image": [], "mask": [], "crs": self.crs, "bbox": query}

        if len(filepaths) == 0:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(filepaths) == 1:
            filenames = filepaths[0]
            query_geom_transformed = None  # is set by the first layer

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)

            for layer in self.layers:

                fn = filenames[layer]

                with rasterio.open(
                    os.path.join(self.root, "enviroatlas_lotp", fn)
                ) as f:
                    dst_crs = f.crs.to_string().lower()

                    if query_geom_transformed is None:
                        query_box_transformed = shapely.ops.transform(
                            self.p_transformers[dst_crs], query_box
                        ).envelope
                        query_geom_transformed = shapely.geometry.mapping(
                            query_box_transformed
                        )

                    data, _ = rasterio.mask.mask(
                        f, [query_geom_transformed], crop=True, all_touched=True
                    )

                if layer in [
                    "naip",
                    "buildings",
                    "roads",
                    "waterways",
                    "waterbodies",
                    "water",
                ]:
                    sample["image"].append(data)
                elif layer in ["prior", "prior_no_osm_no_buildings"]:
                    if self.prior_as_input:
                        sample["image"].append(data)
                    else:
                        sample["mask"].append(data)
                elif layer in ["lc"]:
                    data = self.raw_enviroatlas_to_idx_map[data]
                    sample["mask"].append(data)
        else:
            raise IndexError(f"query: {query} spans multiple tiles which is not valid")

        sample["image"] = np.concatenate(sample["image"], axis=0)
        sample["mask"] = np.concatenate(sample["mask"], axis=0)

        sample["image"] = torch.from_numpy(sample["image"])
        sample["mask"] = torch.from_numpy(sample["mask"])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        def exists(filename: str) -> bool:
            return os.path.exists(os.path.join(self.root, "enviroatlas_lotp", filename))

        if all(map(exists, self.files)):
            return

        # Check if the zip files have already been downloaded
        if os.path.exists(os.path.join(self.root, self.filename)):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(self.url, self.root, filename=self.filename, md5=self.md5)

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.filename))

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Note: only plots the "naip" and "lc" layers.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            ValueError: if the NAIP layer isn't included in ``self.layers``
        """
        if "naip" not in self.layers or "lc" not in self.layers:
            raise ValueError("The 'naip' and 'lc' layers must be included for plotting")

        image_layers = []
        mask_layers = []
        for layer in self.layers:
            if layer in [
                "naip",
                "buildings",
                "roads",
                "waterways",
                "waterbodies",
                "water",
            ]:
                image_layers.append(layer)
            elif layer in ["prior", "prior_no_osm_no_buildings"]:
                if self.prior_as_input:
                    image_layers.append(layer)
                else:
                    mask_layers.append(layer)
            elif layer in ["lc"]:
                mask_layers.append(layer)

        naip_index = image_layers.index("naip")
        lc_index = mask_layers.index("lc")

        image = np.rollaxis(
            sample["image"][naip_index : naip_index + 3, :, :].numpy(), 0, 3
        )
        mask = sample["mask"][lc_index].numpy()

        num_panels = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            predictions = sample["prediction"].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[1].imshow(
            mask, vmin=0, vmax=10, cmap=self.highres_cmap, interpolation="none"
        )
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_predictions:
            axs[2].imshow(
                predictions,
                vmin=0,
                vmax=10,
                cmap=self.highres_cmap,
                interpolation="none",
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Predictions")

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
