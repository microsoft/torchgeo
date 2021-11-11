from torchgeo.datasets import OSCD

ocsd = OSCD(download=True)

sample = ocsd[1]

print(sample["image"].shape)
print(sample["mask"].shape)

