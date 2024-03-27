# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Author: Ian Turton, Glasgow University ian.turton@gla.ac.uk

from io import BytesIO
from typing import Any

import torchvision.transforms as transforms
from owslib.wms import WebMapService
from PIL import Image
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.errors import CRSError
from rtree.index import Index, Property

from torchgeo.datasets import GeoDataset


class WMSDataset(GeoDataset):
    """
    Allow models to fetch images from a WMS (at a good resolution)
    """

    _url = None
    _wms = None

    _layers = []
    _layer = None
    _layer_name = ""
    is_image = True

    def __init__(self, url, res, layer=None, transforms=None, crs=None):
        super().__init__(transforms)
        self._url = url
        self._res = res
        if crs is not None:
            self._crs = CRS.from_epsg(crs)
        self._wms = WebMapService(url)
        self._format = self._wms.getOperationByName("GetMap").formatOptions[0]
        self._layers = list(self._wms.contents)

        if layer in self._layers:
            self.layer(layer, crs)

    def layer(self, layer, crs=None):
        self._layer = self._wms[layer]
        self._layer_name = layer
        coords = self._wms[layer].boundingBox
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        self.index.insert(
            0,
            (
                float(coords[0]),
                float(coords[2]),
                float(coords[1]),
                float(coords[3]),
                0,
                9.223372036854776e18,
            ),
        )
        if crs is None:
            i = 0
            while self._crs is None:
                crs_str = sorted(self._layer.crsOptions)[i].upper()
                if "EPSG:" in crs_str:
                    crs_str = crs_str[5:]
                elif "CRS:84":
                    crs_str = "4326"
                try:
                    self._crs = CRS.from_epsg(crs_str)
                except CRSError:
                    pass

    def getlayer(self):
        return self._layer

    def layers(self):
        return self._layers

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        img = self._wms.getmap(
            layers=[self._layer_name],
            srs="epsg:" + str(self.crs.to_epsg()),
            bbox=(query.minx, query.miny, query.maxx, query.maxy),
            # TODO fix size
            size=(500, 500),
            format=self._format,
            transparent=True,
        )
        sample = {"crs": self.crs, "bbox": query}

        transform = transforms.Compose([transforms.ToTensor()])
        # Convert the PIL image to Torch tensor
        img_tensor = transform(Image.open(BytesIO(img.read())))
        if self.is_image:
            sample["image"] = img_tensor
        else:
            sample["mask"] = img_tensor

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
