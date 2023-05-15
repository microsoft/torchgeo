import time
from tqdm import tqdm

from torchgeo.datasets import L7Irish, random_bbox_assignment, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datamodules import L7IrishDataModule

from torch.utils.data import DataLoader, Dataset, default_collate
from lightning.pytorch import Trainer


if __name__ ==  '__main__':
    print("start loading COGs...")
    start_time = time.time()
    ds = L7Irish(
        root = "/projects/dali/data/l7irish_cog_3857",
        crs = "EPSG:3857",
    )
    sampler = RandomBatchGeoSampler(ds, size=512, batch_size=32, length=10000)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=20, collate_fn=stack_samples)
    for batch in tqdm(dl):
        pass
    print("Dataloader: ", time.time()-start_time)

    #print("Start dataloader and training...")
    #start_time = time.time()

    #dm = L7IrishDataModule(
    #    root = "/projects/dali/data/l7irish_cog_3857",
    #    patch_size=512,
    #    num_workers=20,
    #    length=10000,
    #    )
    #m = SemanticSegmentationTask(
    #    model="unet",
    #    backbone="resnet18",
    #    weights=None,
    #    in_channels=9,
    #    num_classes=5,
    #    loss="ce",
    #    ignore_index=None,
    #    learning_rate=1e-2,
    #    learning_rate_schedule_patience=6,
    #    )
    #trainer = Trainer()
    #trainer.fit(model=m, datamodule=dm)

    #total_time = time.time() - start_time
    #print("Dataloader and training: ", total_time)

