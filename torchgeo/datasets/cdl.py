import glob
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence

import rasterio

from .geo import GeoDataset


class CDL(GeoDataset):
    """The `Cropland Data Layer (CDL)
    <https://data.nal.usda.gov/dataset/cropscape-cropland-data-layer>`_, hosted on
    `CropScape <https://nassgeodata.gmu.edu/CropScape/>`, provides a raster,
    geo-referenced, crop-specific land cover map for the continental United States. The
    CDL also includes a crop mask layer and planting frequency layers, as well as
    boundary, water and road layers. The Boundary Layer options provided are County,
    Agricultural Statistics Districts (ASD), State, and Region. The data is created
    annually using moderate resolution satellite imagery and extensive agricultural
    ground truth.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php#Section1_14.0
    """

    base_folder = "cdl"

    def __init__(
        self,
        root: str = "data",
        bands: Sequence[str] = band_names,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new CDL Dataset.

        Parameters:
            root: root directory where dataset can be found
            bands: bands to return
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = root
        self.bands = bands
        self.transforms = transforms

        fileglob = os.path.join(root, self.base_folder, f"**_{bands[0]}_*.tif")
        for filename in glob.iglob(fileglob):
            # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention
            time = datetime.strptime(
                os.path.basename(filename).split("_")[1], "%Y%m%dT%H%M%S"
            )
            timestamp = time.timestamp()
            with rasterio.open(filename) as f:
                minx, miny, maxx, maxy = f.bounds
                coords = (minx, maxx, miny, maxy, timestamp, timestamp)
                self.index.insert(0, coords, filename)
