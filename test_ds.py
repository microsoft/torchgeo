import time
# from tqdm import tqdm

from torchgeo.datasets import L7Irish, random_bbox_assignment, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler
from torchgeo.datamodules.geo import GeoDataModule
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datamodules import L7IrishDataModule

from torch.utils.data import DataLoader, Dataset, default_collate
# import segmentation_models_pytorch as smp
# import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule, LightningModule, Trainer

ds = L7Irish(
    root = "/projects/dali/data/l7irish_cog",
    crs = "EPSG:3857"
)

# model = smp.Unet(
#     encoder_name="resnet18",
#     encoder_weights="null",
#     in_channels=9,
#     classes=5,
#     loss="ce",
# )

if __name__ ==  '__main__':
    print("start loading COGs...")
    start_time = time.time()
    # sampler = RandomBatchGeoSampler(ds, size=512, batch_size=32)
    # dl = DataLoader(ds, batch_sampler=sampler, num_workers=20, collate_fn=stack_samples)
    dm = L7IrishDataModule(
        ds,
        patch_size=512,
        # batch_size=32,
        num_workers=10,
    )
    m = SemanticSegmentationTask(
        model="unet",
        backbone="resnet18",
        weights=None,
        in_channels=9,
        num_classes=5,
        loss="ce",
        ignore_index=None,
        learning_rate=1e-2,
        learning_rate_schedule_patience=6,
    )
    trainer = Trainer()
    trainer.fit(model=m, datamodule=dm)

    # for batch in tqdm(dl):
        # pass
    total_time = time.time() - start_time
    print("Dataloader for l7irish", total_time)
