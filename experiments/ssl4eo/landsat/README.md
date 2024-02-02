# SSL4EO-L Instructions

This README describes the steps to recreate the datasets and reproduce the results of the SSL4EO-L project.

## Sampling

The first step in creating the SSL4EO-L pre-training and benchmarking datasets is to choose locations from which to sample. The following scripts can be run to choose non-overlapping locations to sample.

```console
$ bash sample_30.sh  # for TM, ETM+, OLI/TIRS
$ bash sample_60.sh  # only for MSS
$ bash sample_conus.sh  # for benchmark datasets
```

The first section of these scripts includes user-specific parameters that can be modified to change the behavior of the scripts. Of particular importance are:

* `SAVE_PATH`: controls where the sampling location CSV is saved to
* `START_INDEX`: index to start from (usually 0, can be increased to append more locations)
* `END_INDEX`: index to stop at (start with ~500K)

These scripts will download world city data and write `sampled_locations.csv` files to be used for downloading.

## Downloading

Next, you'll actually download the data.

```console
$ bash download_mss_raw.sh
$ bash download_tm_toa.sh
$ bash download_etm_toa.sh
$ bash download_etm_sr.sh
$ bash download_oli_tirs_toa.sh
$ bash download_oli_sr.sh
```

These scripts contain the following variables you may want to modify:

* `ROOT_DIR`: root directory containing all subdirectories
* `SAVE_PATH`: where the downloaded data is saved
* `MATCH_FILE`: the CSV created in the previous step
* `NUM_WOKERS`: number of parallel workers
* `START_INDEX`: index from which to start downloading
* `END_INDEX`: index at which to stop downloading

These scripts are designed for downloading the pre-training datasets. Each script can be easily modified to instead download the benchmarking datasets by changing the `MATCH_FILE`, `YEAR`, and `--dates` passed in to the download script. For ETM+ TOA, you'll also want to set a `--default-value` since you'll need to include nodata pixels due to SLC-off.

## Parallel corpus

For each TOA and SR product, we want to create a parallel corpus. This can be done by running:

```console
$ bash delete_mismatch.sh
```

To chop this down to 250K locations, you can then run:

```console
$ bash delete_excess.sh
```

You may want to modify `ROOT_DIR`.

## Compression

The final step in dataset creation is to convert float32 values to uint8 and create compressed COG files. This can be done by running:

```console
$ bash compress_tm_toa.sh
$ bash compress_etm_toa.sh
$ bash compress_etm_sr.sh
$ bash compress_oli_tirs_toa.sh
$ bash compress_oli_sr.sh
```

You may want to modify `ROOT_DIR` or `NUM_WORKERS`.

## Chipping

For the benchmark datasets, there is one additional step required. You should download NLCD and CDL files from the same years as the benchmark datasets, either manually or using TorchGeo. Then you should run:

```console
$ python3 chip_landsat_benchmark.py ...
```

This will create patches of NLCD and CDL data with the same locations and dimensions as the Landsat images you downloaded. Valid options can be found by passing `--help`.

## Running Experiments

Using either the newly created datasets or after downloading the datasets from Hugging Face, you can run each experiment using:

```console
$ torchgeo fit --config *.yaml
```

The config files to be passed can be found in the `conf/` directory. Feel free to tweak any hyperparameters you see in these files. The default values are the optimal hyperparameters we found.

## Plotting

The following scripts can be run to generate the plots in our paper:

```console
$ python3 plot_landsat_bands.py RBV MSS ETM --fig-height=3  # only TM, ETM+, OLI/TIRS
$ python3 plot_landsat_bands.py  # all bands
$ python3 plot_landsat_timeline.py
```
