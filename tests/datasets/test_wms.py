# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from torchgeo.datasets import (
    WMSDataset,
)

import requests
SERVICE_URL = 'https://mesonet.agron.iastate.edu/cgi-bin/wms/nexrad/n0r-t.cgi?'


class TestWMSDataset:

    def service_ok(url, timeout=5):
        try:
            resp = requests.head(url, allow_redirects=True, timeout=timeout)
            ok = resp.ok
        except requests.exceptions.ReadTimeout:
            print('No 2')
            ok = False
        except requests.exceptions.ConnectTimeout:
            print('No 3')
            ok = False
        except Exception:
            print('No 4')
            ok = False
        return ok

    @pytest.mark.online
    @pytest.mark.skipif(not service_ok(SERVICE_URL),
                        reason="WMS service is unreachable")
    def test_wms_no_layer(self):
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0,)
        print(wms.layers())
        assert('nexrad_base_reflect' in wms.layers())
        assert(4326 == wms.crs.to_epsg())
        wms.layer('nexrad_base_reflect', crs=4326)
        assert(-126 == wms.index.bounds[0])
        assert(-66 == wms.index.bounds[1])
        assert(24 == wms.index.bounds[2])
        assert(50 == wms.index.bounds[3])
        assert('image/png' == wms._format)

    def test_wms_layer(self):
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0, layer='nexrad_base_reflect', crs=4326)
        assert(4326 == wms.crs.to_epsg())
        assert(-126 == wms.index.bounds[0])
        assert(-66 == wms.index.bounds[1])
        assert(24 == wms.index.bounds[2])
        assert(50 == wms.index.bounds[3])
        assert('image/png' == wms._format)

    def test_wms_layer_nocrs(self):
        """MESONET GetMap 1.1.1"""
        wms = WMSDataset(SERVICE_URL, 10.0, layer='nexrad_base_reflect')
        assert(4326 == wms.crs.to_epsg())
        assert(-126 == wms.index.bounds[0])
        assert(-66 == wms.index.bounds[1])
        assert(24 == wms.index.bounds[2])
        assert(50 == wms.index.bounds[3])
        assert('image/png' == wms._format)
