# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import urllib3

from torchgeo.datasets import WMSDataset

SERVICE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r-t.cgi?"


def service_ok(url: str, timeout: int = 5) -> bool:
    try:
        http = urllib3.PoolManager()
        resp = http.request("HEAD", url, timeout=timeout)
        ok = 200 == resp.status
    except urllib3.exceptions.NewConnectionError:
        ok = False
    except Exception:
        ok = False
    return ok


class TestWMSDataset:

    @pytest.mark.slow
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

    @pytest.mark.slow
    @pytest.mark.skipif(
        not service_ok(SERVICE_URL), reason="WMS service is unreachable"
    )
    def test_wms_layer(self) -> None:
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0, layer="nexrad_base_reflect", crs=4326)
        assert 4326 == wms.crs.to_epsg()
        assert -126 == wms.index.bounds[0]
        assert -66 == wms.index.bounds[1]
        assert 24 == wms.index.bounds[2]
        assert 50 == wms.index.bounds[3]
        assert "image/png" == wms._format

    @pytest.mark.slow
    @pytest.mark.skipif(
        not service_ok(SERVICE_URL), reason="WMS service is unreachable"
    )
    def test_wms_layer_nocrs(self) -> None:
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0, layer="nexrad_base_reflect")
        assert 4326 == wms.crs.to_epsg()
        assert -126 == wms.index.bounds[0]
        assert -66 == wms.index.bounds[1]
        assert 24 == wms.index.bounds[2]
        assert 50 == wms.index.bounds[3]
        assert "image/png" == wms._format
