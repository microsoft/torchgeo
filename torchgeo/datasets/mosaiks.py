# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .utils import download_url


class MOSAIKS:
    url_prefix = "https://files.codeocean.com/files/verified/fa908bbc-11f9-4421-8bd3-72a4bf00427f_v2.0/data/int/applications/"

    label_urls = {
                "housing": url_prefix + "housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv?download",
                "income": url_prefix + "income/outcomes_sampled_income_CONTUS_16_640_POP_100000_0.csv?download",
                "roads": url_prefix + "roads/outcomes_sampled_roads_CONTUS_16_640_POP_100000_0.csv?download",
                "nightligths": url_prefix + "nightlights/outcomes_sampled_nightlights_CONTUS_16_640_POP_100000_0.csv?download",
                "population": url_prefix + "population/outcomes_sampled_population_CONTUS_16_640_UAR_100000_0.csv?download",
                "elevation": url_prefix + "elevation/outcomes_sampled_elevation_CONTUS_16_640_UAR_100000_0.csv?download",
                "treecover": url_prefix + "treecover/outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv?download",
            }

    def __init__(
        self,
        root: str = "data",
    ) -> None:
        """Initialize a new MOSAIKS dataset instance.
        """

        self.root = root

        self._verify()


    def _verify(self) -> None:
        self._download()

    def _download(self) -> None:
        for f_name in self.label_urls:
            download_url(
                self.label_urls[f_name],
                self.root,
                filename=f_name + ".csv"
            )
