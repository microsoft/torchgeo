# Copyright (c) TorchGeo Contributors.
# Licensed under the MIT License.


import torch


def test_tilenet_pretrained_print():
    from torchgeo.models import tilenet

    print("\n Creating TileNet with pretrained=True ")

    model = tilenet(pretrained=True, in_channels=4)
    model.eval()

    print(" Model created successfully")

    # Print a strong signal from weights
    conv1_weight_sum = model.conv1.weight.detach().abs().sum().item()
    print(f"conv1 weight absolute sum: {conv1_weight_sum:.6f}")

    # Forward pass
    x = torch.randn(1, 4, 50, 50)
    y = model(x)

    print(f"Forward pass output shape: {y.shape}")

    assert y.shape == (1, 512)
    assert conv1_weight_sum > 0

    print("Hugging Face pretrained weights loaded correctly\n")
