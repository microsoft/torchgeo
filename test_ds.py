import time
from tqdm import tqdm

from torchgeo.datasets import L7Irish, random_bbox_assignment, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.datamodules.geo import GeoDataModule

from torch.utils.data import DataLoader, Dataset, default_collate

ds = L7Irish(
    root = "/Users/yc/projects/dali/data/l7irish",
    crs = "EPSG:3857"
)

sampler = RandomBatchGeoSampler(ds, size=256, batch_size=32, length=32*150)
dl = DataLoader(ds, batch_sampler=sampler, num_workers=10, collate_fn=stack_samples)

if __name__ ==  '__main__':
    start_time = time.time()
    for batch in tqdm(dl):
        pass
    print("Total time: ", time.time()-start_time)