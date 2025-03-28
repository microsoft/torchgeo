"""Copernicus-Pretrain dataset."""

import random
from typing import Any, ClassVar

import webdataset as wds


class CopernicusPretrain:
    """Copernicus-Pretrain dataset.

    Copernicus-Pretrain is an extension of the SSL4EO-S12 dataset to all major Sentinel missions (S1-S5P).
    The images are organized into ~310K regional grids (0.25°x0.25°, consistent with ERA5), densely covering
    the whole land surface and near-land ocean with time series from eight distinct Sentinel modalities.

    This dataset class uses WebDataset for efficient data loading in distributed environments, which returns a
    Pytorch IterableDataset that is compatible with Pytorch DataLoader. Note: it is recommended to further use
    webdataset.WebLoader (a wrapper on DataLoader) for more features in data loading.

    The full dataset has varying number of modalities, S1/2 local patches, and timestamps for different grids.
    For simplicity, the current dataset class provides a minimum example:
    - only use grids with all modalities (220k)
    - sample one local patch for S1 and S2
    - sample one timestamp for each modality
    Therefore, each sample contains 8 tensors (S1, S2, S3, S5P NO2/CO/SO2/O3, DEM) and a JSON metadata.

    Example:
    ```
    copernicus_pretrain = CopernicusPretrain(shards_path='data/example-{000000..000009}.tar', shuffle=100, shardshuffle=True, resampled=True)
    train_dataset = copernicus_pretrain.get_webdataset()

    ## check the first sample
    for sample in train_dataset:
        s1, s2, s3, s5p_co, s5p_no2, s5p_o3, s5p_so2, dem, meta = sample
        break

    ## create a DataLoader for distributed training on 2 GPUs
    train_dataset = train_dataset.batched(10) # batch size
    train_loader = webdataset.WebLoader(train_dataset, batch_size=None, num_workers=2)
    # unbatch, shuffle, and rebatch to mix samples from different workers
    train_loader = train_loader.unbatched().shuffle(100).batched(10)
    # A resampled dataset is infinite size, but we can recreate a fixed epoch length
    number_of_batches = 1000 // (10 * 2) # total number of samples / (batch size * world size)
    data_loader_train = data_loader_train.with_epoch(number_of_batches)
    ```

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849

    .. versionadded:: 0.7
    """

    urls: ClassVar[dict[str, str]] = {
        '220k_aligned': 'https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain/resolve/main/ssl4eo_s_220k_aligned/example-{000000..002255}.tar',  # grids with all modalities
        '220k_310k_union': 'https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain/resolve/main/ssl4eo_s_220k_310k_union/example-{002256..003210}.tar',  # remaining grids (with at least one modality)
        '100_example': 'https://huggingface.co/datasets/wangyi111/Copernicus-Pretrain/resolve/main/example_100_grids/example_100_webdataset/example-{000000..000009}.tar',  # 100 example grids
    }

    def __init__(
        self,
        shards_path: str,
        shuffle: int = 0,
        shardshuffle: bool = False,
        resampled: bool = False,
    ) -> None:
        """Initialize a new CopernicusPretrain instance.

        Args:
            shards_path (str): Path to the shards of the dataset. Can be local paths or URLs.
            resampled (bool): Dynamically resample the dataset shards.
            shardshuffle (bool): Shuffle the order of the shards.
            shuffle (int): Buffer size for shuffling individual samples before batching.
        """
        self.shards_path = shards_path
        self.shuffle = shuffle
        self.shardshuffle = shardshuffle
        self.resampled = resampled

    def has_all_modalities(self, sample: dict[str, Any]) -> bool:
        """Mapping function: filter samples with all required modalities."""
        required_keys = [
            's1_grd.pth',
            's2_toa.pth',
            's3_olci.pth',
            's5p_co.pth',
            's5p_no2.pth',
            's5p_o3.pth',
            's5p_so2.pth',
            'dem.pth',
            'json',
        ]
        return all(key in sample for key in required_keys)

    def sample_one_local_patch(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Mapping function: randomly select one local patch for S1 and S2."""
        s1, s2 = sample['s1_grd.pth'], sample['s2_toa.pth']
        meta_s1, meta_s2 = sample['json']['s1_grd'], sample['json']['s2_toa']

        idx = random.randint(0, s1.shape[0] - 1)
        sample['s1_grd.pth'], sample['s2_toa.pth'] = s1[idx], s2[idx]
        sample['json']['s1_grd'], sample['json']['s2_toa'] = meta_s1[idx], meta_s2[idx]
        return sample

    def sample_one_time_stamp(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Mapping function: randomly select one timestamp for all modalities."""
        for key in sample:
            if key.endswith('.pth') and key != 'dem.pth':
                idx = random.randint(0, sample[key].shape[0] - 1)
                sample[key] = sample[key][idx]
                sample['json'][key.replace('.pth', '')] = sample['json'][
                    key.replace('.pth', '')
                ][idx]

        sample['json']['dem'] = sample['json']['dem'][0]
        return sample

    def get_webdataset(self) -> wds.WebDataset:
        """Creates an IterableDataset using WebDataset."""
        dataset = (
            wds.WebDataset(
                self.shards_path,
                resampled=self.resampled,
                shardshuffle=self.shardshuffle,
                nodesplitter=wds.split_by_node,
            )  # shuffle shard orders and samples within shards, split by node
            .shuffle(self.shuffle)  # shuffle individual samples before batching
            .decode()  # decode binary data
            .select(self.has_all_modalities)  # select samples with all modalities
            .map(self.sample_one_local_patch)  # sample one local patch for S1 and S2
            .map(self.sample_one_time_stamp)  # sample one timestamp for all modalities
            .to_tuple(
                's1_grd.pth',  # 2x264x264
                's2_toa.pth',  # 13x264x264
                's3_olci.pth',  # 21x96x96
                's5p_co.pth',  # 1x28x28
                's5p_no2.pth',  # 1x28x28
                's5p_o3.pth',  # 1x28x28
                's5p_so2.pth',  # 1x28x28
                'dem.pth',  # 1x960x960
                'json',  # metadata
            )  # convert to tuple
        )

        return dataset
