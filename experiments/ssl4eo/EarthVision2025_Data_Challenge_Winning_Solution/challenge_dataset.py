import torch
from torch.utils.data import Dataset
import os
import glob
import xarray as xr
import numpy as np
from typing import List, Dict


# Mean and standard devation for the challenge data.
# Note that these are different from the SSL4EO-S12 v1.1 moments.
S1GRD_MEAN = [-11.834, -19.243]
S1GRD_STD = [4.305, 5.479]

S2L1C_MEAN = [1635.299, 1402.885, 1289.505, 1281.272, 1534.981, 2272.474, 2630.972, 2587.956, 2889.274, 976.031, 20.369, 2109.307, 1350.051]
S2L1C_STD = [1123.963, 1187.2, 1128.715, 1322.882, 1285.925, 1250.079, 1325.492, 1294.318, 1343.536, 636.232, 27.82, 1023.855, 834.773]

S2L2A_MEAN = [802.067, 917.472, 1130.01, 1210.515, 1587.985, 2355.781, 2650.339, 2787.571, 2860.466, 2921.651, 2221.172, 1549.952]
S2L2A_STD = [1563.348, 1604.422, 1553.908, 1622.652, 1593.07, 1523.264, 1556.785, 1618.961, 1532.417, 1653.532, 1183.218, 1025.306]


# SSL4EO-S12 v1.1 mean and standard deviation. Recommended if shift_s2_channels = True.
S2L1C_MEAN_SSL4EO = [2607.345, 2393.068, 2320.225, 2373.963, 2562.536, 3110.071, 3392.832, 3321.154, 3583.77, 1838.712, 1021.753, 3205.112, 2545.798]
S2L1C_STD_SSL4EO = [786.523, 849.702, 875.318, 1143.578, 1126.248, 1161.98, 1273.505, 1246.79, 1342.755, 576.795, 45.626, 1340.347, 1145.036]

S2L2A_MEAN_SSL4EO = [1793.243, 1924.863, 2184.553, 2340.936, 2671.402, 3240.082, 3468.412, 3563.244, 3627.704, 3711.071, 3416.714, 2849.625]
S2L2A_STD_SSL4EO = [1160.144, 1201.092, 1219.943, 1397.225, 1400.035, 1373.136, 1429.17, 1485.025, 1447.836, 1652.703, 1471.002, 1365.307]

S1GRD_MEAN_SSL4EO = [-12.577, -20.265]
S1GRD_STD_SSL4EO = [5.179, 5.872]


class E2SChallengeDataset(Dataset):

    def __init__(self, 
                 data_path: str = None, 
                 transform = None, 
                 modalities: List[str] = None,
                 dataset_name: str = 'bands', 
                 seasons: int = 4, 
                 randomize_seasons: bool = False,
                 concat: bool = True,
                 output_file_name: bool = False,
                 shift_s2_channels: bool = True
                ):
        """Dataset class for the embed2scale challenge data

        Parameters
        ----------
        data_path : str, path-like
            Path to challenge data. Assumes that under data_path there are 3 subfolders, named after the modalities.
        transform : torch.Compose
            Transformations to apply to the data
        modalities : list[str]
            List of modalities to include. Should correpond to the subfolders under data_path.
        dataset_name : str
            Name of dataset in zarr archive. Use 'bands' here. Defaults to 'bands'.
        seasons : int
            Number of seasons to load. Must be integer between 1 and 4. Default is 4.
        randomize_seasons : bool
            Toggle randomized order of seasons. If True, the order of the seasons will be randomized. Default is False.
        concat : bool
            Toggle concatenating the modalities along the channel dimension. Default is True.
        output_file_name : bool
            Toggle output of the file name.
        shift_s2_channels : bool
            Toggle shifting the S2 channels by 1000 to align to SSL4EO-S12 v1.1. Default is True, where the challenge data S2 channels are 
            shifted upward 1000 to have the range as SSL4EO-S12 v1.1. The background is that ESA decided 
            from 2022-01-25 to shift the DN values of S2 by 1000 upward. SSL4EO-S12 v1.1 includes this shift, 
            while the challenge data does not.

        Returns
        -------
        torch.Tensor or dict
            If output_file_name=False, outputs a torch.Tensor. 
            If output_file_name=True, outputs a dictionary with fields 'data' and 'file_name'. 'data' is a torch.Tensor if concat=True and a dict with one field per modality, each containing a torch.Tensor if False. 'file_name' is the id of the loaded file.
        """

        self.data_path = data_path
        self.transform = transform
        self.modalities = modalities
        self.dataset_name = dataset_name
        assert isinstance(seasons, int) and (1 <= seasons <= 4), "Number of seasons must be integer between 1 and 4."
        
        self.seasons = seasons
        self.randomize_seasons = randomize_seasons
        if not randomize_seasons:
            self.possible_seasons = list(range(seasons))
        else:
            self.possible_seasons = list(range(4))
        assert len(modalities) > 0, "No modalities provided."
        self.concat = concat
        self.output_file_name = output_file_name
        self.shift_s2_channels = shift_s2_channels
        
        self.samples = glob.glob(os.path.join(data_path, modalities[0], '*.zarr.zip'))
        

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample_path = self.samples[idx]
        file_name = os.path.splitext(os.path.basename(sample_path))[0].replace('.zarr', '')
        if self.randomize_seasons:
            seasons = [self.possible_seasons[ind] for ind in torch.randperm(len(self.possible_seasons)).tolist()[:self.seasons]]
        else:
            seasons = self.possible_seasons
        sample_paths = [sample_path] + [sample_path.replace(self.modalities[0], modality) for modality in self.modalities[1:]]
        data = {}
        
        for modality, sample_path in zip(self.modalities, sample_paths):
            season_index = xr.DataArray(seasons, dims='time')
            data[modality] = xr.open_zarr(sample_path).isel(time=season_index)[self.dataset_name].values

            # Add shift to modality, typically used to align S2 channels with SSL4EO-S12 v1.1
            if self.shift_s2_channels and (modality in ['s2l1c', 's2l2a']):
                data[modality] += 1000

        n_bands_per_modality = {m: d.shape[-3] for m, d in data.items()}
        start_ind_of_modality = {m: n for m, n in zip(self.modalities, [0] + np.cumsum(list(n_bands_per_modality.values())).tolist())}

        # Concatenate data
        data = np.concatenate(list(data.values()), axis=-3)
        data = data.astype(np.float32) # uint16 before, but that type is not accepted by from_numpy()
        data = torch.from_numpy(data)
        
        # Transform
        if self.transform is not None:
            data = self.transform(data)
            
        if not self.concat:
            data = {m: data[..., start_ind_of_modality[m]: start_ind_of_modality[m] + n_bands_per_modality[m], :, :] for m in self.modalities}

        if self.output_file_name:
            return {'data': data, 'file_name': file_name}
        else:
            return data


def collate_fn(batch):
    if isinstance(batch, dict) or isinstance(batch, torch.Tensor):
        # Single sample
        return batch
    elif isinstance(batch, list) and isinstance(batch[0], torch.Tensor):
        # Concatenate tensors along sample dim
        return torch.concat(batch, dim=0)
    elif isinstance(batch, list) and isinstance(batch[0], dict):
        file_names = [sample['file_name'] for sample in batch]
        data = [sample['data'] for sample in batch]
        if isinstance(data[0], torch.Tensor):
            data = torch.concat(data, dim=0)
        elif isinstance(data[0], dict):
            data = {
                m: torch.concat([b[m] for b in data], dim=0)
                for m in data[0].keys()
            }
        return {'data': data, 'file_name': file_names}
    