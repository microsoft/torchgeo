# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import requests

from torchgeo.datasets import WMSDataset

SERVICE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r-t.cgi?"


def service_ok(url: str, timeout: int = 5) -> bool:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        ok = bool(resp.ok)
    except requests.exceptions.ReadTimeout:
        ok = False
    except requests.exceptions.ConnectTimeout:
        ok = False
    except Exception:
        ok = False
    return ok


class TestWMSDataset:

    @pytest.mark.online
    @pytest.mark.skipif(
        not service_ok(SERVICE_URL), reason="WMS service is unreachable"
    )
    def test_wms_no_layer(self) -> None:
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0)
        assert "nexrad_base_reflect" in wms.layers()
        assert 4326 == wms.crs.to_epsg()
        wms.layer("nexrad_base_reflect", crs=4326)
        assert -126 == wms.index.bounds[0]
        assert -66 == wms.index.bounds[1]
        assert 24 == wms.index.bounds[2]
        assert 50 == wms.index.bounds[3]
        assert "image/png" == wms._format

    def test_wms_layer(self) -> None:
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0, layer="nexrad_base_reflect", crs=4326)
        assert 4326 == wms.crs.to_epsg()
        assert -126 == wms.index.bounds[0]
        assert -66 == wms.index.bounds[1]
        assert 24 == wms.index.bounds[2]
        assert 50 == wms.index.bounds[3]
        assert "image/png" == wms._format

    def test_wms_layer_nocrs(self) -> None:
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0, layer="nexrad_base_reflect")
        assert 4326 == wms.crs.to_epsg()
        assert -126 == wms.index.bounds[0]
        assert -66 == wms.index.bounds[1]
        assert 24 == wms.index.bounds[2]
        assert 50 == wms.index.bounds[3]
        assert "image/png" == wms._format
