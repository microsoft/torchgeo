import time
from tqdm import tqdm

from torchgeo.datasets import L7Irish, random_bbox_assignment, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.datamodules.geo import GeoDataModule

from torch.utils.data import DataLoader, Dataset, default_collate

#ds1 = L7Irish(
#    root = "/projects/dali/data/l7irish",
#    crs = "EPSG:3857"
#)
ds2 = L7Irish(
    root = "/projects/dali/data/l7irish_cog",
    crs = "EPSG:3857"
)
#ds3 = L7Irish(
#    root = "/projects/dali/data/l7irish_trans",
#    crs = "EPSG:3857"
#)
#ds4 = L7Irish(
#    root = "/projects/dali/data/l7irish_trans2",
#    crs = "EPSG:3857"
#)
# dss = [ds1, ds2, ds3, ds4]
#dss = [ds2]
# avg_time = []

if __name__ ==  '__main__':
    print("start loading COGs...")
    start_time = time.time()
    sampler = RandomBatchGeoSampler(ds2, size=512, batch_size=32)
    dl = DataLoader(ds2, batch_sampler=sampler, num_workers=40, collate_fn=stack_samples)
    for batch in tqdm(dl):
        pass
    total_time = time.time() - start_time
    print("Dataloader for l7irish", total_time)
