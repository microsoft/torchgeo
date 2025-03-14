from torch.utils.data import Dataset

from .senbench_cloud_s3 import SenBenchCloudS3

# Dictionary to map dataset names to classes
DATASET_REGISTRY = {
    'senbench-cloud-s3': SenBenchCloudS3
    # To add other datasets
}


class SentinelBench(Dataset):
    """Wrapper to dynamically load a dataset from SentinelBench."""

    def __init__(self, dataset_name: str, **kwargs):
        """Args:
        dataset_name (str): The name of the dataset to load.
        **kwargs: All other arguments will be forwarded to the dataset's __init__ method.
        """
        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{dataset_name}' not found! Available datasets: {list(DATASET_REGISTRY.keys())}"
            )

        # Get the dataset class from registry
        dataset_class: type[Dataset] = DATASET_REGISTRY[dataset_name]

        # Dynamically initialize the dataset with the given arguments
        self.dataset = dataset_class(**kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
