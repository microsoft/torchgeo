# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Open Buildings datasets."""

import glob
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import rasterio
import shapely
import shapely.wkt as wkt
import torch
from packaging.version import parse
from rasterio.crs import CRS
from rtree.index import Index, Property

from .geo import VectorDataset
from .utils import BoundingBox, check_integrity


class OpenBuildings(VectorDataset):
    r"""Open Buildings dataset.

    The `Open Buildings
    <https://sites.research.google/open-buildings/#download>`__ dataset
    consists of computer generated building detections across the African continent.

    Dataset features:

    * 516M building detections as polygons with centroid lat/long
    * covering area of 19.4M km\ :sup:`2`\  (64% of the African continent)
    * confidence score and
      `Plus Code <https://maps.google.com/pluscodes/>`_

    Dataset format:

    * csv files containing building detections compressed as csv.gz
    * meta data geojson file

    The data can be downloaded from `here
    <https://sites.research.google/open-buildings/#download>`__. Additionally, the
    `meta data geometry file
    <https://sites.research.google/open-buildings/tiles.geojson>`_ also needs to be
    placed in `root` as `tiles.geojson`.

    If you use this dataset in your research, please cite the following technical
    report:

    * https://arxiv.org/abs/2107.12283

    .. versionadded:: 0.3
    """

    md5s = {
        "025_buildings.csv.gz": "41db2572bfd08628d01475a2ee1a2f17",
        "04f_buildings.csv.gz": "3232c1c6d45c1543260b77e5689fc8b1",
        "05b_buildings.csv.gz": "4fc57c63bbbf9a21a3902da7adc3a670",
        "093_buildings.csv.gz": "00fce146dadf0b30255e750c4c5ac2de",
        "095_buildings.csv.gz": "f5765b0936f7ccbd0b4abed60d994f08",
        "0c3_buildings.csv.gz": "013b130fe872387e0cff842399b423de",
        "0c3_buildings.csv": "a697ad2433e9a9f6001de25b4664651a",
        "0c5_buildings.csv.gz": "16ca283e9344e9da8b47acaf03c1c6e4",
        "0c7_buildings.csv.gz": "b3774930006497a80c8a2fbf33056610",
        "0d1_buildings.csv.gz": "41e652218ca5964d297d9cd1d84b831c",
        "0d7_buildings.csv.gz": "d365fe47d10b0756dd54ceca24598d8e",
        "0d9_buildings.csv.gz": "3ebd47fa4f86857266e9a7346d6aa163",
        "0db_buildings.csv.gz": "368213e9caa7ee229ef9403b0ca8c80d",
        "0dd_buildings.csv.gz": "8f5fcefff262fdfd82800092d2e9d841",
        "0df_buildings.csv.gz": "cbb5f63b10daa25568bdde8d9f66f8a4",
        "0e1_buildings.csv.gz": "a9b9bf1e541b62c8a34d2f6f2ae71e1c",
        "0e3_buildings.csv.gz": "3d9c2ffc11c02aec2bd008699f9c4bd1",
        "0e5_buildings.csv.gz": "1e1b2bf63dfc520e62e4b68db23fe64c",
        "0e7_buildings.csv.gz": "c96797588c90e66268367cb56b4b9af8",
        "0e9_buildings.csv.gz": "c53bb7bbc8140034d1be2c49ff49af68",
        "0eb_buildings.csv.gz": "407c771f614a15d69d78f1e25decf694",
        "0ed_buildings.csv.gz": "bddd10992d291677019d7106ce1f4fac",
        "0ef_buildings.csv.gz": "d1b91936e7ac06c661878ef9eb5dba7b",
        "0f1_buildings.csv.gz": "9d86eb10d2d8766e1385b6c52c11d5e2",
        "0f9_buildings.csv.gz": "1c6775131214b26f4a27b4c42d6e9fca",
        "0fb_buildings.csv.gz": "d39528cb4e0cbff589ca89dc86d9b5db",
        "0fd_buildings.csv.gz": "304fe4a60e950c900697d975098f7536",
        "0ff_buildings.csv.gz": "266ca7ed1ad0251b3999b0e2e9b54648",
        "103_buildings.csv.gz": "8d3cafab5f1e02b2a0a6180eb34d1cac",
        "105_buildings.csv.gz": "dd61cc74239aa9a1b30f10859122807b",
        "107_buildings.csv.gz": "823c05984f859a1bf17af8ce78bf2892",
        "109_buildings.csv.gz": "cfdee0e807168cd1c183d9c01535369b",
        "10b_buildings.csv.gz": "d8ecaf406abd864b641ba34985f3042e",
        "10d_buildings.csv.gz": "af584a542a17942ff7e94653322dba87",
        "10f_buildings.csv.gz": "3d5369e15c4d1f59fb38cf61f4e6290b",
        "111_buildings.csv.gz": "47504e43d1b67101bed5d924225328dc",
        "113_buildings.csv.gz": "3f991c831569f91f34eaa8fc3882b2fd",
        "117_buildings.csv.gz": "a4145fa6e458480e30c807f80ae5cd65",
        "119_buildings.csv.gz": "5661b7ac23f266542c7e0d962a8cae58",
        "11b_buildings.csv.gz": "41b6d036610d0bddac069ec72e68710e",
        "11d_buildings.csv.gz": "1ef75e9d176dd8d6bfa6012d36b1d25c",
        "11f_buildings.csv.gz": "f004873d1ef3933c1716ab6409565b7d",
        "121_buildings.csv.gz": "0c7e7a9043ed069fbdefdcfcfc437482",
        "123_buildings.csv.gz": "c46bd53b67025c3de11657220cce0aec",
        "125_buildings.csv.gz": "33253ae1a82656f4eedca9bd86f981a3",
        "127_buildings.csv.gz": "2f827f8fc93485572178e9ad0c65e22d",
        "129_buildings.csv.gz": "74f98346990a1d1e41241ce8f4bb201a",
        "12f_buildings.csv.gz": "b1b0777296df2bfef512df0945ca3e14",
        "131_buildings.csv.gz": "8362825b10c9396ecbb85c49cd210bc6",
        "137_buildings.csv.gz": "96da7389df820405b0010db4a6c98c61",
        "139_buildings.csv.gz": "c41e26fc6f3565c3d7c66ab977dc8159",
        "13b_buildings.csv.gz": "981d4ccb0f41a103bdad8ef949eb4ffe",
        "13d_buildings.csv.gz": "d15585d06ee74b0095842dd887197035",
        "141_buildings.csv.gz": "ae0bf17778d45119c74e50e06a04020d",
        "143_buildings.csv.gz": "9699809e57eb097dfaf9d484f1d9c5fa",
        "145_buildings.csv.gz": "81e74e0165ea358278ce18507dddfdb0",
        "147_buildings.csv.gz": "39edad15fa16c432f5d460f0a8166032",
        "149_buildings.csv.gz": "94bf8f8fa221744fb1d57c7d4065e69e",
        "14f_buildings.csv.gz": "ca8410be89b5cf868c2a67861712e4ea",
        "15b_buildings.csv.gz": "8c0071c0ae20a60e8dd4d7aa6aac5a99",
        "15d_buildings.csv.gz": "35f044a323556adda5f31e8fc9307c85",
        "161_buildings.csv.gz": "ba08b70a26f07b5e2cd4eafd9d6f826b",
        "163_buildings.csv.gz": "2bec83a2504b531cd1cb0311fcb6c952",
        "165_buildings.csv.gz": "48f934733dd3054164f9b09abee63312",
        "167_buildings.csv.gz": "bba8657024d80d44e475759b65adc969",
        "169_buildings.csv.gz": "13e142e48597ee7a8b0b812e226dfa72",
        "16b_buildings.csv.gz": "9c62351d6cc8eaf761ab89d4586d26d6",
        "16d_buildings.csv.gz": "a33c23da3f603c8c3eacc5e6a47aaf66",
        "16f_buildings.csv.gz": "4850dd7c8f0fb628ba5864ea9f47647b",
        "171_buildings.csv.gz": "4217f1b025db869c8bed1014704c2a79",
        "173_buildings.csv.gz": "5a5f3f07e261a9dc58c6180b69130e4a",
        "175_buildings.csv.gz": "5bbf7a7c8f57d28e024ddf8f4039b575",
        "177_buildings.csv.gz": "76cd4b17d68d62e1f088f229b65f8acf",
        "179_buildings.csv.gz": "a5a1c6609483336ddff91b2385e70eb9",
        "17b_buildings.csv.gz": "a47c1145a3b0bcdaba18c153b7b92b87",
        "17d_buildings.csv.gz": "3226d0abf396f44c1a436be83538dfd8",
        "17f_buildings.csv.gz": "3e18d4fc5837ee89274d30f2126b92b2",
        "181_buildings.csv.gz": "c87639d7f6d6a85a3fa6b06910b0e145",
        "183_buildings.csv.gz": "e94438ebf19b3b25035954d23a0e90cf",
        "185_buildings.csv.gz": "8de8d1d50c16c575f85b96dee474cb56",
        "189_buildings.csv.gz": "da94cd495a99496fd687bbb4a1715c90",
        "18b_buildings.csv.gz": "9ab353335fe6ff694e834889be2b305d",
        "18d_buildings.csv.gz": "e37e0f868ce96f7d14f7bf1a301da1d3",
        "18f_buildings.csv.gz": "e9000b9ef9bb0f838088e96becfc95a1",
        "191_buildings.csv.gz": "c00bb4d6b2b12615d576c06fe545cbfa",
        "193_buildings.csv.gz": "d48d4c03ef053f6987b3e6e9e78a8b03",
        "195_buildings.csv.gz": "d93ab833e74480f07a5ccf227067db5a",
        "197_buildings.csv.gz": "8667e040f9863e43924aafe6071fabc7",
        "199_buildings.csv.gz": "04ba65a4caf16cc1e0d5c4e1322c5885",
        "19b_buildings.csv.gz": "e49412e3e1bccceb0bdb4df5201288f4",
        "19d_buildings.csv.gz": "92b5fb4e96529d90e99c788e3e8696d4",
        "19f_buildings.csv.gz": "c023f6c37d0026b56f530b841517a6cd",
        "1a1_buildings.csv.gz": "471483b50c722af104af8a582e780c04",
        "1a3_buildings.csv.gz": "0a453053f1ff53f9e165e16c7f97354a",
        "1a5_buildings.csv.gz": "1f6a823e223d5f29c66aa728933de684",
        "1a7_buildings.csv.gz": "6130b724501fa16e6d84e484c4091f1f",
        "1a9_buildings.csv.gz": "73022e8e7b994e76a58cc763a057d542",
        "1b9_buildings.csv.gz": "48dea4af9d12b755e75b76c68c47de6b",
        "1bb_buildings.csv.gz": "dfb9ee4d3843d81722b70f7582c775a4",
        "1bd_buildings.csv.gz": "fdea2898fc50ae25b6196048373d8244",
        "1bf_buildings.csv.gz": "96ef27d6128d0bcdfa896fed6f27cdd0",
        "1c1_buildings.csv.gz": "32e3667d939e7f95316eb75a6ffdb603",
        "1c3_buildings.csv.gz": "ed94b543da1bbe3101ed66f7d7727d24",
        "1c5_buildings.csv.gz": "ce527ab33e564f0cc1b63ae467932a18",
        "1c7_buildings.csv.gz": "d5fb474466d6a11d3b08e3a011984ada",
        "1dd_buildings.csv.gz": "9e7e50e3f95b3f2ceff6351b75ca1e75",
        "1e5_buildings.csv.gz": "f95ea85fce47ce7edf5729086d43f922",
        "1e7_buildings.csv.gz": "2bca5682c48134e69b738d70dfe7d516",
        "1e9_buildings.csv.gz": "f049ad06dbbb200f524b4f50d1df8c2e",
        "1eb_buildings.csv.gz": "6822d7f202b453ec3cc03fb8f04691ad",
        "1ed_buildings.csv.gz": "9dfc560e2c3d135ebdcd46fa09c47169",
        "1ef_buildings.csv.gz": "506e7772c35b09cfd3b6f8691dc2947d",
        "1f1_buildings.csv.gz": "b74f2b585cfad3b881fe4f124080440a",
        "1f3_buildings.csv.gz": "12896642315320e11ed9ed2d3f0e5995",
        "1f5_buildings.csv.gz": "334aea21e532e178bf5c54d028158906",
        "1f7_buildings.csv.gz": "0e8c3d2e005eb04c6852a8aa993f5a76",
        "217_buildings.csv.gz": "296e9ba121fea752b865a48e5c0fe8a5",
        "219_buildings.csv.gz": "1d19b6626d738f7706f75c2935aaaff4",
        "21d_buildings.csv.gz": "28bfca1f8668f59db021d3a195994768",
        "21f_buildings.csv.gz": "06325c8b0a8f6ed598b7dc6f0bb5adf2",
        "221_buildings.csv.gz": "a354ffc1f7226d525c7cf53848975da1",
        "223_buildings.csv.gz": "3bda1339d561b3bc749220877f1384d9",
        "225_buildings.csv.gz": "8eb02ad77919d9e551138a14d3ad1bbc",
        "227_buildings.csv.gz": "c07aceb7c81f83a653810befa0695b61",
        "22f_buildings.csv.gz": "97d63e30e008ec4424f6b0641b75377c",
        "231_buildings.csv.gz": "f4bc384ed74552ddcfe2e69107b91345",
        "233_buildings.csv.gz": "081756e7bdcfdc2aee9114c4cfe62bd8",
        "23b_buildings.csv.gz": "75776d3dcbc90cf3a596664747880134",
        "23d_buildings.csv.gz": "e5d0b9b7b14601f58cfdb9ea170e9520",
        "23f_buildings.csv.gz": "77f38466419b4d391be8e4f05207fdf5",
        "3d1_buildings.csv.gz": "6659c97bd765250b0dee4b1b7ff583a9",
        "3d5_buildings.csv.gz": "c27d8f6b2808549606f00bc04d8b42bc",
        "3d7_buildings.csv.gz": "abdef2e68cc31c67dbb6e60c4c40483e",
        "3d9_buildings.csv.gz": "4c06ae37d8e76626345a52a32f989de9",
        "3db_buildings.csv.gz": "e83ca0115eaf4ec0a72aaf932b00442a",
        "b5b_buildings.csv.gz": "5e5f59cb17b81137d89c4bab8107e837",
    }

    filename_glob = "*_buildings.csv"
    zipfile_glob = "*_buildings.csv.gz"

    meta_data_url = "https://sites.research.google/open-buildings/tiles.geojson"
    meta_data_filename = "tiles.geojson"

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: float = 0.0001,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.root = root
        self.res = res
        self.checksum = checksum
        self.root = root
        self.res = res
        self.transforms = transforms

        self._verify()

        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )

        # Create an R-tree to index the dataset using the polygon centroid as bounds
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        with open(os.path.join(root, "tiles.geojson")) as f:
            data = json.load(f)

        features = data["features"]
        features_filenames = [
            feature["properties"]["tile_url"].split("/")[-1] for feature in features
        ]  # get csv filename

        polygon_files = glob.glob(os.path.join(self.root, self.zipfile_glob))
        polygon_filenames = [f.split(os.sep)[-1] for f in polygon_files]

        matched_features = [
            feature
            for filename, feature in zip(features_filenames, features)
            if filename in polygon_filenames
        ]

        i = 0
        source_crs = CRS.from_dict({"init": "epsg:4326"})
        for feature in matched_features:
            if crs is None:
                crs = CRS.from_dict(source_crs)

            c = feature["geometry"]["coordinates"][0]
            xs = [x[0] for x in c]
            ys = [x[1] for x in c]

            minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)

            (minx, maxx), (miny, maxy) = fiona.transform.transform(
                source_crs.to_dict(), crs.to_dict(), [minx, maxx], [miny, maxy]
            )
            mint = 0
            maxt = sys.maxsize
            coords = (minx, maxx, miny, maxy, mint, maxt)

            filepath = os.path.join(
                self.root, feature["properties"]["tile_url"].split("/")[-1]
            )
            self.index.insert(i, coords, filepath)
            i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{self.root}'"
            )

        self._crs = crs
        self._source_crs = source_crs

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata for the given query. If there are
            not matching shapes found within the query, an empty raster is returned

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        shapes = self._filter_geometries(query, filepaths)

        # Rasterize geometries
        width = (query.maxx - query.minx) / self.res
        height = (query.maxy - query.miny) / self.res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if shapes:
            masks = rasterio.features.rasterize(
                shapes, out_shape=(int(height), int(width)), transform=transform
            )
            masks = torch.tensor(masks).unsqueeze(0)
        else:
            masks = torch.zeros(size=(1, int(height), int(width)))

        sample = {"mask": masks, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _filter_geometries(
        self, query: BoundingBox, filepaths: List[str]
    ) -> List[Dict[str, Any]]:
        """Filters a df read from the polygon csv file based on query and conf thresh.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            filepaths: filepaths to files that were hits from rmtree index

        Returns:
            List with all polygons from all hit filepaths

        """
        import pandas as pd

        # We need to know the bounding box of the query in the source CRS
        (minx, maxx), (miny, maxy) = fiona.transform.transform(
            self._crs.to_dict(),
            self._source_crs.to_dict(),
            [query.minx, query.maxx],
            [query.miny, query.maxy],
        )
        df_query = (
            "longitude >= {} & longitude <= {} & " "latitude >= {} & latitude <= {}"
        ).format(minx, maxx, miny, maxy)
        shapes = []
        for f in filepaths:
            csv_chunks = pd.read_csv(f, chunksize=200000, compression="gzip")
            for chunk in csv_chunks:
                df = chunk.query(df_query)
                # Warp geometries to requested CRS
                polygon_series = df["geometry"].map(self._wkt_fiona_geom_transform)
                shapes.extend(polygon_series.values.tolist())

        return shapes

    def _wkt_fiona_geom_transform(self, x: str) -> Dict[str, Any]:
        """Function to transform a geometry string into new crs.

        Args:
            x: Polygon string

        Returns:
            transformed geometry in geojson format

        """
        x = json.dumps(shapely.geometry.mapping(wkt.loads(x)))
        x = json.loads(x.replace("'", '"'))
        import fiona

        if parse(fiona.__version__) >= parse("1.9a1"):
            import fiona.model

            geom = fiona.model.Geometry(**x)
        else:
            geom = x
        transformed: Dict[str, Any] = fiona.transform.transform_geom(
            self._source_crs.to_dict(), self._crs.to_dict(), geom
        )
        return transformed

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing or checksum fails
            FileNotFoundError: if metadata file is not found in root
        """
        # Check if the zip files have already been downloaded and checksum
        pathname = os.path.join(self.root, self.zipfile_glob)
        i = 0
        for zipfile in glob.iglob(pathname):
            filename = os.path.basename(zipfile)
            if self.checksum and not check_integrity(zipfile, self.md5s[filename]):
                raise RuntimeError(f"Dataset found, but corrupted: {filename}.")
            i += 1

        if i != 0:
            return

        # check if the metadata file has been downloaded
        if not os.path.exists(os.path.join(self.root, self.meta_data_filename)):
            raise FileNotFoundError(
                f"Meta data file {self.meta_data_filename} "
                f"not found in in `root={self.root}`."
            )

        raise RuntimeError(
            f"Dataset not found in `root={self.root}` "
            "either specify a different `root` directory or make sure you "
            "have manually downloaded the dataset as suggested in the documentation."
        )

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample["mask"].permute(1, 2, 0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].permute(1, 2, 0)
            ncols = 2
        else:
            ncols = 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
