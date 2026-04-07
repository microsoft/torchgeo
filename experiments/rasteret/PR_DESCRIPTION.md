## Summary

This PR adds `torchgeo.datasets.RasteretDataset`, an experimental opt-in dataset for Rasteret-backed COG collections.

## Intended Mental Model

The intended usage is:

1. Build or load a Rasteret `Collection`
2. Construct `RasteretDataset(collection=..., bands=...)`
3. Continue with standard TorchGeo usage

`RasteretDataset` is a TorchGeo-facing wrapper over Rasteret's existing
`Collection.to_torchgeo_dataset(...)` integration.

After dataset construction, normal TorchGeo workflows stay the same:
- `RandomGeoSampler`
- `GridGeoSampler`
- dataloaders and collation
- transforms
- dataset composition

The difference is primarily **how raster scenes are prepared and read**, not how TorchGeo training code is written downstream.

## Usage

```python
import rasteret
from torchgeo.datasets import RasteretDataset
from torchgeo.samplers import RandomGeoSampler

collection = rasteret.build(
    "earthsearch/sentinel-2-l2a",
    name="s2-bangalore",
    bbox=(77.5, 12.9, 77.7, 13.1),
    date_range=("2024-01-01", "2024-06-30"),
)

# Or directly load prebuilt Collections, in this case Google AEF COGs from SourceCoop
# collection = rasteret.load("aef/v1-annual")

dataset = RasteretDataset(
    collection=collection,
    bands=["B04", "B03", "B02"],
)

sampler = RandomGeoSampler(dataset, size=256, length=100)
```

## Ease Of Use

For an existing TorchGeo workflow, the integration is intentionally small in surface area:

- build or load one Rasteret `Collection` (`build(...)`, `build_from_stac(...)`,
  `build_from_table(...)`, or `load(...)`)
- swap dataset construction to `RasteretDataset(collection=..., bands=...)`
- keep TorchGeo samplers, dataloaders, transforms, and training code unchanged

## Benchmark

The benchmark compares:
- the same set of data
- the same band
- the same chip size and batch size
- the same output CRS / resolution
- native TorchGeo `RasterDataset` vs `RasteretDataset`

The default `bangalore` preset auto-builds Collection if missing.

## Benchmark Numbers

Measured with:
```bash
python experiments/rasteret/benchmark.py --collection bangalore --mode both --sampler both
```

Environment:
- Python `3.13.11`
- TorchGeo `0.10.0.dev0`
- Rasteret `0.3.7`
- Rasterio `1.4.4`

Results (`B04`, `res=(10.0, 10.0)`, `chip=256`, `batch=4`):

 | Scenario | Speedup | Rasteret | TorchGeo |
  |---|---:|---:|---:|
  | spatial/random | 27.7x | 1369 ms | 37878 ms |
  | spatial/grid | 25.5x | 451 ms | 11518 ms |
  | timeseries/random | 12.7x | 2874 ms | 36437 ms |
  | timeseries/grid | 5.2x | 1184 ms | 6097 ms |

Additional benchmark scenarios are documented in Rasteret's published
[TorchGeo comparison notebook](https://terrafloww.github.io/rasteret/tutorials/05_torchgeo_comparison/):

Scenario | rasterio/GDAL | Rasteret | Speedup
-- | -- | -- | --
Single AOI, 15 scenes | 9.08 s | 1.14 s | 8x
Multi-AOI, 30 scenes | 42.05 s | 2.25 s | 19x
Cross-CRS boundary, 12 scenes | 12.47 s | 0.59 s | 21x


---


## What This PR Does

- adds `torchgeo.datasets.RasteretDataset`
- adds targeted tests for dataset construction and wrapper behavior
- adds TorchGeo-facing API docs
- adds a benchmark script comparing native TorchGeo/rasterio reads with
  Rasteret-backed TorchGeo reads

## What This PR Does Not Do

This PR does **not**:

- replace `RasterDataset`
- replace rasterio/GDAL across TorchGeo wholesale
- auto-convert existing TorchGeo raster datasets into Rasteret datasets
- redesign TorchGeo's STAC direction

This is intentionally an experimental, opt-in dataset entry point.

## Relation To Ongoing TorchGeo Discussions

This PR is relevant to several existing TorchGeo discussions, but does not try to settle them broadly:

- `#403` STAC API dataset
- `#2382` Time Series Support
- `#3160` backend discussions around optional experimental data access paths

Rasteret already supports building `Collections` from:
- Rasteret registry datasets via `build(...)`
- STAC APIs via `build_from_stac(...)`
- Parquet / GeoParquet tables with COG URLs via `build_from_table(...)`

Collections can be exported and reloaded with `export()` / `load(...)`.

## Why A Separate Dataset Class Exists

TorchGeo's native `RasterDataset` implementation is generally path-first: it
discovers scenes from file paths, builds its own index, and reads pixels
through rasterio/GDAL.

Rasteret is collection-first: scene metadata is prepared ahead of time in a
`Collection`, and that collection is the starting point for filtering and read
planning.

`RasteretDataset` exists to bridge those two models cleanly:

- **TorchGeo-facing side:** a first-class raster dataset entry point with
  familiar parameters like `bands`, `crs`, `res`, and `time_series`
- **Rasteret-facing side:** delegates indexing and COG read execution to
  Rasteret from an existing `Collection`

This keeps downstream TorchGeo usage familiar without forcing Rasteret into
TorchGeo's path-scanning implementation.

## What A Rasteret Collection Is

A Rasteret `Collection` is a queryable table of raster scene metadata backed in
memory by a `pyarrow.dataset.Dataset`, typically persisted as Parquet or
GeoParquet.

In practice, the `Collection` is where scene discovery and filtering happen
first, and `RasteretDataset` is the TorchGeo dataset surface created from that
prepared collection.

## Dependency / Packaging Notes

This PR adds Rasteret as an optional extra and updates CI to install it during relevant jobs.

Rasteret currently constrains rasterio to `<1.5.0`, so the lockfile resolves to `rasterio==1.4.4` when the Rasteret extra is included.

## Test Coverage

Targeted tests:

```bash
pytest tests/datasets/test_rasteret.py
```
