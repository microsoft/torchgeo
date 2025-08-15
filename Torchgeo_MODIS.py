# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torchgeo.datasets import RasterDataset
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
import pystac_client
import planetary_computer
from rioxarray.exceptions import NoDataInBounds
from typing import Optional


PRODUCTS = {
    "10A1": {
        "collection": "modis-10A1-061",
        "preferred_asset_keys": [
            "NDSI_Snow_Cover",
            "Snow_Albedo_Daily_Tile",
            "NDSI_Snow_Cover_Basic_QA",
            "NDSI_Snow_Cover_Algorithm_Flags_QA",
        ],
        "name": "Snow Cover Daily",
        "bands": [
            "NDSI_Snow_Cover",
            "Snow_Albedo_Daily_Tile",
            "NDSI_Snow_Cover_Basic_QA",
            "NDSI_Snow_Cover_Algorithm_Flags_QA",
        ],
        "vmin": 0,
        "vmax": 100,
    },
    "64A1": {
        "collection": "modis-64A1-061",
        "preferred_asset_keys": ["QA", "Burn_Date"],
        "name": "Burned Area Monthly",
        "bands": ["FireMask"],
        "vmin": 0,
        "vmax": 1,
    },
    "11A1": {
        "collection": "modis-11A1-061",
        "preferred_asset_keys": ["LST_Day_1km", "LST"],
        "name": "Land Surface Temperature/Emissivity Daily",
        "bands": [
            "LST_Day_1km",
            "Clear_day_cov",
            "Day_view_angl",
            "Day_view_time",
            "LST_Night_1km",
            "Clear_night_cov",
            "Night_view_angl",
            "Night_view_time",
        ],
        "vmin": 250,
        "vmax": 325,
    },
    "13A1": {
        "collection": "modis-13A1-061",
        "preferred_asset_keys": ["NDVI", "EVI"],
        "name": "Vegetation Indices (NDVI/EVI)",
        "bands": [
            "500m_16_days_EVI",
            "500m_16_days_NDVI",
            "500m_16_days_VI_Quality",
            "500m_16_days_MIR_reflectance",
            "500m_16_days_NIR_reflectance",
            "500m_16_days_red_reflectance",
            "500m_16_days_blue_reflectance",
            "500m_16_days_sun_zenith_angle",
            "500m_16_days_pixel_reliability",
            "500m_16_days_view_zenith_angle",
            "500m_16_days_relative_azimuth_angle",
            "500m_16_days_composite_day_of_the_year",
        ],
        "vmin": 0,
        "vmax": 0,
    },
    "14A2": {
        "collection": "modis-14A2-061",
        "preferred_asset_keys": ["FireMask", "QA"],
        "name": "Thermal Anomalies/Fire Daily",
        "bands": ["FireMask", "QA"],
        "vmin": 0,
        "vmax": 10,
    },
    "15A2H": {
        "collection": "modis-15A2H-061",
        "preferred_asset_keys": ["LeafAreaIndex"],
        "name": "Leaf Area Index",
        "bands": [
            "500m_16_days_EVI",
            "500m_16_days_NDVI",
            "500m_16_days_VI_Quality",
            "500m_16_days_MIR_reflectance",
            "500m_16_days_NIR_reflectance",
            "500m_16_days_red_reflectance",
            "500m_16_days_blue_reflectance",
            "500m_16_days_sun_zenith_angle",
            "500m_16_days_pixel_reliability",
            "500m_16_days_view_zenith_angle",
            "500m_16_days_relative_azimuth_angle",
            "500m_16_days_composite_day_of_the_year",
        ],
        "vmin": 0,
        "vmax": 10,
    },
    "43A4": {
        "collection": "modis-43A4-061",
        "preferred_asset_keys": ["NBAR"],
        "name": "Nadir BRDF-Adjusted Reflectance (NBAR) Daily",
        "bands": [
            "BRDF_Albedo_Band_Mandatory_Quality_Band1",
            "BRDF_Albedo_Band_Mandatory_Quality_Band2",
            "BRDF_Albedo_Band_Mandatory_Quality_Band3",
            "BRDF_Albedo_Band_Mandatory_Quality_Band4",
            "BRDF_Albedo_Band_Mandatory_Quality_Band5",
            "BRDF_Albedo_Band_Mandatory_Quality_Band6",
            "BRDF_Albedo_Band_Mandatory_Quality_Band7",
        ],
        "vmin": 1,
        "vmax": 3000,
    },
}


class TorchGeoMODISRasterDataset(RasterDataset):
    paths = ["dummy"]

    @property
    def resources(self):
        return ["dummy"]

    def _check_existence(self) -> bool:
        return True

    def _check_exists(self) -> bool:
        return True

    def download(self):
        pass

    def __init__(
        self,
        item_id: str,
        product_code: str,
        bbox: Optional[tuple] = None,
        transforms: Optional[callable] = None,
    ):
        object.__init__(self)
        self.transforms = transforms
        self.item_id = item_id
        self.product_code = product_code
        self.bbox = bbox

        if product_code not in PRODUCTS:
            raise ValueError(f"Product code '{product_code}' not found in PRODUCTS.")
        product_info = PRODUCTS[product_code]
        self.collection = product_info["collection"]
        self.preferred_asset_keys = product_info["preferred_asset_keys"]
        self.product_name = product_info["name"]
        self.desired_bands = product_info.get("bands", [])
        self.vmin = product_info.get("vmin", None)
        self.vmax = product_info.get("vmax", None)

        print(f"Loading MODIS {self.product_name} ({self.product_code}) for item {self.item_id}")

    
        client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = client.search(collections=[self.collection], ids=[self.item_id])
        items = list(search.items())
        if not items:
            raise RuntimeError(f"Item {self.item_id} not found in collection {self.collection}!")
        self.item = items[0]
        print("Found item:", self.item.id)
        print("Available asset keys:", list(self.item.assets.keys()))

        self.asset_urls = {}
        for band in self.desired_bands:
            if band in self.item.assets:
                asset = self.item.assets[band]
                self.asset_urls[band] = planetary_computer.sign(asset.href)
            else:
                for key in self.preferred_asset_keys:
                    if key in self.item.assets:
                        asset = self.item.assets[key]
                        self.asset_urls[band] = planetary_computer.sign(asset.href)
                        break
                else:
                    print(f"Warning: No asset found for desired band '{band}'.")

        self.ds = self.load_data()

    def load_data(self) -> xr.DataArray:
        band_arrays = []
        band_names = []
        for band in self.desired_bands:
            if band in self.asset_urls:
                try:
                    da = rioxarray.open_rasterio(self.asset_urls[band], masked=True)
                    if self.bbox is not None:
                        try:
                            da = da.rio.clip_box(*self.bbox)
                        except NoDataInBounds:
                            print(f"Warning: Provided bbox does not intersect data for band {band}.")
                    if "band" in da.dims and da.sizes["band"] == 1:
                        da = da.squeeze("band", drop=True)
                    da = da.expand_dims(dim="band")
                    band_arrays.append(da)
                    band_names.append(band)
                    print(f"Loaded band {band} with shape {da.shape}")
                except Exception as e:
                    print(f"Error loading band {band}: {e}")
            else:
                print(f"Warning: Band '{band}' not found in asset URLs.")
        if not band_arrays:
            raise RuntimeError("No bands could be loaded!")
        ds = xr.concat(band_arrays, dim="band")
        ds = ds.assign_coords(band=band_names)
        if self.transforms:
            ds = self.transforms(ds)
        print("Combined dataset shape:", ds.shape)
        return ds

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        array = self.ds.data
        if hasattr(array, "compute"):
            array = array.compute()
        image = torch.from_numpy(array.astype("float32"))
        sample = {
            "image": image,
            "asset_urls": self.asset_urls,
            "product_code": self.product_code,
            "product_name": self.product_name,
            "bands": list(self.ds.coords["band"].values) if "band" in self.ds.coords else [],
            "bounds": self.ds.rio.bounds() if hasattr(self.ds, "rio") else None,
        }
        return sample

    def plot(self, cmap="inferno"):
        """Plot the first band."""
        if "band" in self.ds.dims and self.ds.sizes["band"] > 1:
            vis_data = self.ds.isel(band=0)
        else:
            vis_data = self.ds
        vis_data = vis_data.squeeze()
        try:
            bounds = self.ds.rio.bounds()
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        except Exception:
            extent = None

        plt.figure(figsize=(10, 8))
        im = plt.imshow(vis_data, cmap=cmap, extent=extent, vmin=self.vmin, vmax=self.vmax)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        band_label = self.ds.coords["band"].values[0] if "band" in self.ds.coords else "Band 0"
        plt.title(f"MODIS {self.product_name} - Band: {band_label}")
        plt.colorbar(im, label="Pixel Value")
        plt.show()

    def plot_all_bands(self, cmap="inferno"):
        """Plot all bands in separate subplots."""
        bands_list = list(self.ds.coords["band"].values) if "band" in self.ds.coords else ["Band 0"]
        n_bands = self.ds.sizes["band"] if "band" in self.ds.dims else 1

        fig, axes = plt.subplots(n_bands, 1, figsize=(10, 5 * n_bands))
        if n_bands == 1:
            axes = [axes]
        try:
            bounds = self.ds.rio.bounds()
            extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        except Exception:
            extent = None

        for i in range(n_bands):
            ax = axes[i]
            band_name = bands_list[i] if i < len(bands_list) else f"Band {i+1}"
            im = ax.imshow(self.ds.isel(band=i), cmap=cmap, extent=extent, vmin=self.vmin, vmax=self.vmax)
            ax.set_title(f"MODIS {self.product_name} - {band_name}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="Pixel Value")
        plt.tight_layout()
        plt.show()


#%% Testing 

if __name__ == "__main__":
    
    # MODIS Burned Area Monthly
    burned_item_id = "MCD64A1.A2024275.h09v04.061.2024347115015"
    dataset = TorchGeoMODISRasterDataset(item_id=burned_item_id, product_code="64A1", bbox=None)
    sample = dataset[0]
    print("Sample image tensor shape:", sample["image"].shape)
    print("Bands:", sample["bands"])
    dataset.plot(cmap="Reds")
    
    # MODIS Snow Cover
    snow_item_id = "MYD10A1.A2025023.h09v04.061.2025025033315"
    snow_dataset = TorchGeoMODISRasterDataset(item_id=snow_item_id, product_code="10A1", bbox=None)
    sample_snow = snow_dataset[0]
    print("TorchGeo MODIS Snow Cover sample image tensor shape:", sample_snow["image"].shape)
    print("Bands (from product settings):", sample_snow["bands"])
    snow_dataset.plot(cmap="winter")
    
    # MODIS Land Surface Temperature (LST)
    lst_item_id = "MYD11A1.A2025032.h21v06.061.2025035030644"
    lst_dataset = TorchGeoMODISRasterDataset(item_id=lst_item_id, product_code="11A1", bbox=None)
    sample_lst = lst_dataset[0]
    print("MODIS LST sample image tensor shape:", sample_lst["image"].shape)
    print("Bands (from product settings):", sample_lst["bands"])
    lst_dataset.plot(cmap="magma")

    
    # MODIS Vegetation Indices
    veg_item_id = "MYD13A1.A2025009.h24v05.061.2025030153936"
    veg_dataset = TorchGeoMODISRasterDataset(item_id=veg_item_id, product_code="13A1", bbox=None)
    sample_veg = veg_dataset[0]
    print("MODIS Vegetation Indices sample image tensor shape:", sample_veg["image"].shape)
    print("Bands (from product settings):", sample_veg["bands"])
    veg_dataset.plot(cmap="Greens")

    
    # MODIS Thermal Anomalies (8-Day)
    ther8_item_id = "MYD14A2.A2025017.h18v03.061.2025030145023"
    ther8_dataset = TorchGeoMODISRasterDataset(item_id=ther8_item_id, product_code="14A2", bbox=None)
    sample_ther8 = ther8_dataset[0]
    print("MODIS Thermal Anomalies 8-Day sample image tensor shape:", sample_ther8["image"].shape)
    print("Bands (from product settings):", sample_ther8["bands"])
    ther8_dataset.plot(cmap="YlOrRd")

    
    # MODIS Nadir BRDF-Adjusted Reflectance (NBAR)
    nbar_item_id = "MCD43A4.A2025023.h10v06.061.2025034221006"
    nbar_dataset = TorchGeoMODISRasterDataset(item_id=nbar_item_id, product_code="43A4", bbox=None)
    sample_nbar = nbar_dataset[0]
    print("MODIS NBAR sample image tensor shape:", sample_nbar["image"].shape)
    print("Bands (from product settings):", sample_nbar["bands"])
    nbar_dataset.plot(cmap="viridis")


    
    