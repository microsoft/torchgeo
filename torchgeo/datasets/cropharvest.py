# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CropHarvest datasets."""

import glob
import json
import os
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import DatasetNotFoundError, download_url, extract_archive


class CropHarvest(NonGeoDataset):
    """CropHarvest dataset.

    `CropHarvest <https://github.com/nasaharvest/cropharvest>`_ is a
    crop classification dataset.

    Dataset features:

    * single pixel timeseries with croptype labels
    * 18 bands per image over 12 months

    Dataset format:

    * images are 12x18 ndarrays with 18 bands over 12 months

    Dataset properties:

    1. is_crop - whether or not a single pixel contains cropland
    2. classification_label - optional field identifying a specific croptype
    3. dataset - source dataset for the imagery
    4. lat
    5. lon

    If you use this dataset in your research, please cite the following paper:

    * https://openreview.net/forum?id=JtjzUXPEaCu

    This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.6
    """

    # *https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/bands.py
    all_bands = [
        "VV",
        "VH",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
        "temperature_2m",
        "total_precipitation",
        "elevation",
        "slope",
        "NDVI",
    ]
    rgb_bands = ["B4", "B3", "B2"]

    classes = [
        "None",
        "Other",
        "cerrado",
        "pasture",
        "coffee",
        "uncultivated soil",
        "hay",
        "conversion area",
        "eucalyptus",
        "brachiaria",
        "cotton",
        "wheat",
        "rice",
        "orchard",
        "alfalfa",
        "maize",
        "vineyard",
        "cassava",
        "sorghum",
        "bean",
        "groundnut",
        "fallowland",
        "millet",
        "tomato",
        "sugarcane",
        "sweetpotato",
        "banana",
        "soybean",
        "cabbage",
        "safflower",
        "dry bean",
        "bush bean",
        "yellow maize",
        "white sorghum",
        "sunflower",
        "palm",
        "potatoes",
        "groundnuts or peanuts",
        "barley",
        "pulses",
        "blé tendre d\x92hiver",
        "maïs",
        "jachère de 6 ans ou plus déclarée comme surface d\x92intérêt écologique",
        "prairie permanente - herbe prédominante (ressources fourragères ligneuses absentes ou peu présentes)",  # noqa: E501
        "orge de printemps",
        "jachère de 6 ans ou plus",
        "orge d'hiver",
        "colza d\x92hiver",
        "pomme de terre de consommation",
        "jachère de 5 ans ou moins",
        "autre trèfle",
        "autre graminée fourragère pure de 5 ans ou moins",
        "autre fourrage annuel d\x92un autre genre",
        "betterave non fourragère / bette",
        "autre luzerne",
        "lin fibres",
        "mélange de légumineuses prépondérantes au semis et de graminées fourragères de 5 ans ou moins",  # noqa: E501
        "\x8cillette",
        "pois de printemps semé avant le 31/05",
        "triticale d\x92hiver",
        "soja",
        "avoine d\x92hiver",
        "verger",
        "autre prairie temporaire de 5 ans ou moins",
        "oignon / échalote",
        "haricot / flageolet",
        "pois d\x92hiver",
        "laitue / batavia / feuille de chêne",
        "chanvre",
        "bois pâturé",
        "avoine de printemps",
        "prairie en rotation longue (6 ans ou plus)",
        "féverole semée avant le 31/05",
        "blé tendre de printemps",
        "sorgho",
        "tournesol",
        "sarrasin",
        "mélange de légumineuses fourragères prépondérantes et de céréales et/ou d\x92oléagineux",  # noqa: E501
        "maïs ensilage",
        "coriandre",
        "mélange de céréales",
        "blé dur d\x92hiver",
        "autre légume ou fruit annuel",
        "épeautre",
        "mélange de protéagineux (pois et/ou lupin et/ou féverole) prépondérants semés avant le 31/05 et de céréales",  # noqa: E501
        "carotte",
        "autre légume ou fruit pérenne",
        "petit fruit rouge",
        "pois chiche",
        "ray-grass de 5 ans ou moins",
        "petits pois",
        "surface pastorale - herbe prédominante et ressources fourragères ligneuses présentes",  # noqa: E501
        "seigle d\x92hiver",
        "autre pois fourrager de printemps",
        "autre vesce",
        "luzerne déshydratée",
        "lentille cultivée (non fourragère)",
        "autre pois fourrager d\x92hiver",
        "betterave fourragère",
        "autre sainfoin",
        "moha",
        "fraise",
        "autre céréale d\x92un autre genre",
        "lin non textile d\x92hiver",
        "épinard",
        "persil",
        "dactyle de 5 ans ou moins",
        "autre plante à parfum, aromatique et médicinale annuelle",
        "trèfle déshydraté",
        "menthe",
        "fétuque de 5 ans ou moins",
        "potiron / potimarron",
        "autre plante à parfum, aromatique et médicinale pérenne",
        "lin non textile de printemps",
        "vigne : raisins de cuve non en production",
        "pomme de terre féculière",
        "mélange de légumineuses fourragères (entre elles)",
        "moutarde",
        "courge musquée / butternut",
        "cerfeuil",
        "seigle de printemps",
        "triticale de printemps",
        "colza de printemps",
        "culture sous serre hors sol",
        "radis",
        "navette d\x92été",
        "lentille fourragère",
        "estragon",
        "oseille",
        "camomille",
        "cresson",
        "aubergine",
        "tomate",
        "thym",
        "blé dur de printemps",
        "autre plante fourragère sarclée d\x92un autre genre",
        "ciboulette",
        "basilic",
        "fourrage composé de céréales et/ou de protéagineux (en proportion <\xa050%) et/ou de légumineuses fourragères (en proportion < 50%)",  # noqa: E501
        "maïs doux",
        "lotier",
        "fenouil",
        "autre céréale d\x92hiver de genre triticum",
        "noix",
        "vigne : raisins de cuve",
        "vesce déshydratée",
        "noisette",
        "courgette / citrouille",
        "autre oléagineux d\x92un autre genre",
        "céleri",
        "chou",
        "navet",
        "panais",
        "chicorée / endive / scarole",
        "poivron / piment",
        "aïl",
        "chanvre sans étiquette conforme",
        "poireau",
        "marjolaine / origan",
        "mélange de protéagineux (pois et/ou lupin et/ou féverole)",
        "topinambour",
        "fève",
        "cameline",
        "autre féverole fourragère",
        "aneth",
        "gesse",
        "fenugrec",
        "mélange d\x92oléagineux",
        "concombre / cornichon",
        "houblon",
        "autre céréale de genre panicum",
        "lupin doux de printemps semé avant le 31/05",
        "sarriette",
        "sauge",
        "artichaut ",
        "lavande / lavandin",
        "melon",
        "autre mélilot",
        "mâche",
        "plantain psyllium",
        "cerise bigarreau pour transformation",
        "surface pastorale - ressources fourragères ligneuses prédominantes",
        "pastèque",
        "autre céréale de printemps de genre zea",
        "canne à sucre - propriété ou faire valoir direct",
        "canne à sucre - fermage",
        "agrume",
        "verger (dom)",
        "légume sous abri",
        "banane créole (fruit et légume) - autre",
        "ananas",
        "plante aromatique (autre que vanille)",
        "vanille sous bois",
        "café / cacao",
        "curcuma",
        "géranium",
        "vanille",
        "banane créole (fruit et légume) - fermage",
        "plante médicinale",
        "horticulture ornementale de plein champ",
        "horticulture ornementale sous abri",
        "banane créole (fruit et légume) - propriété ou faire valoir direct",
        "autre céréale de genre fagopyrum",
        "ylang-ylang",
        "canne à sucre - autre",
        "canne à sucre - indivision",
        "banane créole (fruit et légume) - indivision",
        "avocat",
        "plante à parfum (autre que géranium et vétiver)",
        "banane créole (fruit et légume) - réforme foncière",
        "arachide ",
        "vigne : raisins de table",
        "vanille verte",
        "canne à sucre - réforme foncière",
        "jachère noire",
        "banane export - fermage",
        "banane export - propriété ou faire valoir direct",
        "banane export - autre",
        "banane export - indivision",
        "prune d\x92ente pour transformation",
        "wetland",
        "abandoned (overgrown)",
        "urban",
        "shrubland",
        "mixed forage",
        "pasture (undiff)",
        "sod",
        "cereals (undiff)",
        "barley (undiff)",
        "corn",
        "timothy",
        "spring wheat",
        "soybeans",
        "forest",
        "nursery",
        "winter barley",
        "canola/rapeseed",
        "winter wheat",
        "unimproved pasture",
        "christmas trees",
        "greenhouse",
        "winter rye",
        "pasture/forage",
        "red clover",
        "abandoned (shrubs)",
        "other",
        "fallow",
        "native grassland",
        "barren",
        "water",
        "oats",
        "coniferous",
        "white clover",
        "broadleaf",
        "clover (undiff)",
        "sunflowers",
        "greenfeed (mixed)",
        "beans (undiff)",
        "apples",
        "vegetables (undiff)",
        "raspberry",
        "fruits (berry & annual)",
        "strawberry",
        "onions",
        "broccoli",
        "cauliflower",
        "peas (undiff)",
        "field peas",
        "sugarbeets",
        "ginseng",
        "rye (undiff)",
        "cherries",
        "cucumber",
        "tomatoes",
        "zucchini",
        "carrot",
        "tobacco",
        "asparagus",
        "other agriculture (undiff)",
        "lettuce",
        "pumpkin/squash",
        "brussel sprout",
        "lavendar",
        "sweet corn",
        "peaches",
        "pepper",
        "pears",
        "miscanthus",
        "fruits (trees)",
        "blueberry - high bush",
        "fababeans",
        "mixedwood",
        "hops",
        "buckwheat",
        "spring rye",
        "too wet to seed",
        "blueberry (undiff)",
        "triticale (undiff)",
        "ryegrass",
        "flaxseed",
        "birdsfoot trefoil",
        "ornemental",
        "plums",
        "apricots",
        "spring barley",
        "sweet clover",
        "wheat (undiff)",
        "fescue",
        "beetroot",
        "spring triticale",
        "wheatgrass",
        "eggplant",
        "blueberry - low bush",
        "cranberry",
        "mustard",
        "hemp",
        "blackberry",
        "dill",
        "herbs",
        "cedar",
        "hazelnut",
        "raddish",
        "nuts",
        "deforested",
        "winter cereals",
        "other nursery",
        "oilseeds (undiff)",
        "pulses (undiff)",
        "winter triticale",
        "poplar",
        "celery",
        "hascap",
        "walnut",
        "other berries",
        "peatland - harvested",
        "other nut",
        "spelt",
        "biofuel (undiff)",
        "artichoke",
        "other forage",
        "vetch",
        "lentils",
        "quinoa",
        "other orchards",
        "forage crops",
        "meadows",
        "rye",
        "oil seeds",
        "root crops",
        "groundnuts",
        "sesame",
    ]

    features_url = "https://zenodo.org/records/7257688/files/features.tar.gz?download=1"
    labels_url = "https://zenodo.org/records/7257688/files/labels.geojson?download=1"
    file_dict = {
        "features": {
            "url": features_url,
            "filename": "features.tar.gz",
            "extracted_filename": os.path.join("features", "arrays"),
            "md5": "cad4df655c75caac805a80435e46ee3e",
        },
        "labels": {
            "url": labels_url,
            "filename": "labels.geojson",
            "extracted_filename": "labels.geojson",
            "md5": "bf7bae6812fc7213481aff6a2e34517d",
        },
    }

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CropHarvest dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: if ``download=False`` but
                dataset is missing or checksum fails
        """
        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download

        self._verify()

        self.files = self._load_features(self.root)
        self.labels = self._load_labels(self.root)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            single pixel timeseries array and label at that index
        """
        files = self.files[index]
        data = self._load_array(files["chip"])

        label = self._load_label(files["index"], files["dataset"])
        sample = {"array": data, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_features(self, root: str) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing path for each of hd5 single pixel time series and
            its key for associated data
        """
        files = []
        chips = glob.glob(
            os.path.join(root, self.file_dict["features"]["extracted_filename"], "*.h5")
        )
        chips = sorted(os.path.basename(chip) for chip in chips)
        for chip in chips:
            chip_path = os.path.join(
                root, self.file_dict["features"]["extracted_filename"], chip
            )
            index = chip.split("_")[0]
            dataset = chip.split("_")[1][:-3]
            files.append(dict(chip=chip_path, index=index, dataset=dataset))
        return files

    def _load_labels(self, root: str) -> pd.DataFrame:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            pandas dataframe containing label data for each feature
        """
        filename = self.file_dict["labels"]["extracted_filename"]
        with open(os.path.join(root, filename), encoding="utf8") as f:
            data = json.load(f)

            pd.json_normalize(data["features"])
            df = pd.json_normalize(data["features"])

            return df

    def _load_array(self, path: str) -> Tensor:
        """Load an individual single pixel timeseries.

        Args:
            path: path to the image

        Returns:
            the image
            ImportError if h5py is not installed
        """
        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError(
                "h5py is not installed and is required to use this dataset"
            )
        filename = os.path.join(path)
        with h5py.File(filename, "r") as f:
            array = f.get("array")[()]
            tensor = torch.from_numpy(array).float()
            return tensor

    def _load_label(self, idx: str, dataset: str) -> Tensor:
        """Load the croptype label for a single pixel timeseries.

        Args:
            idx: sample index in labels.geojson
            dataset: dataset name to query labels.geojson

        Returns:
            the crop type label
        """
        index = int(idx)
        row = self.labels[
            (self.labels["properties.index"] == index)
            & (self.labels["properties.dataset"] == dataset)
        ]
        row = row.to_dict(orient="records")[0]
        label = "None"
        if row["properties.label"]:
            label = row["properties.label"]
        elif row["properties.is_crop"] == 1:
            label = "Other"

        return torch.tensor(self.classes.index(label))

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        # Check if feature files already exist
        feature_path = os.path.join(
            self.root, self.file_dict["features"]["extracted_filename"]
        )
        feature_path_zip = os.path.join(
            self.root, self.file_dict["features"]["filename"]
        )
        label_path = os.path.join(
            self.root, self.file_dict["labels"]["extracted_filename"]
        )
        # Check if labels exist
        if os.path.exists(label_path):
            # Check if features exist
            if os.path.exists(feature_path):
                return
            # Check if features are downloaded in zip format
            if os.path.exists(feature_path_zip):
                self._extract()
                return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        features_path = os.path.join(self.file_dict["features"]["filename"])
        download_url(
            self.file_dict["features"]["url"],
            self.root,
            filename=features_path,
            md5=self.file_dict["features"]["md5"] if self.checksum else None,
        )

        download_url(
            self.file_dict["labels"]["url"],
            self.root,
            filename=os.path.join(self.file_dict["labels"]["filename"]),
            md5=self.file_dict["labels"]["md5"] if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        features_path = os.path.join(self.root, self.file_dict["features"]["filename"])
        extract_archive(features_path)

    def plot(self, sample: dict[str, Tensor], subtitle: Optional[str] = None) -> Figure:
        """Plot a sample from the dataset using bands for Agriculture RGB composite.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional subtitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, axs = plt.subplots()
        bands = [self.all_bands.index(band) for band in self.rgb_bands]
        rgb = np.array(sample["array"])[:, bands]
        normalized = rgb / np.max(rgb, axis=1, keepdims=True)
        axs.imshow(normalized[None, ...])
        axs.set_title(f'Croptype: {self.classes[sample["label"]]}')

        if subtitle is not None:
            plt.suptitle(subtitle)

        return fig
