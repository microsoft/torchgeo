# CD Model Benchmark Results on OSCD100

Config: lr=0.0001, loss=bce, in_channels=3 (RGB), max_epochs=50, patience=10

## Summary

| Model           | Backbone              | Weights           | F1         | IoU        | Accuracy   | Epochs | Time (min) |
| --------------- | --------------------- | ----------------- | ---------- | ---------- | ---------- | ------ | ---------- |
| **btc**         | swin_tiny             | True (Cityscapes) | **0.5416** | **0.3714** | **0.9690** | 26     | 0.6        |
| fcsiamdiff      | resnet18              | False             | 0.4986     | 0.3321     | 0.9489     | 50     | 0.6        |
| unet (baseline) | resnet18              | False             | 0.4957     | 0.3295     | 0.9597     | 50     | ~2.0       |
| fcsiamdiff      | resnet18              | True (ImageNet)   | 0.0089     | 0.0045     | 0.9567     | 50     | 1.1        |
| fcsiamconc      | resnet18              | False             | 0.0000     | 0.0000     | 0.9473     | 50     | 0.8        |
| fcsiamconc      | resnet18              | True (ImageNet)   | 0.0000     | 0.0000     | 0.9374     | 50     | 0.7        |
| changevit       | vit_small_patch16_224 | True (ImageNet)   | 0.0000     | 0.0000     | 0.9462     | 14     | 0.7        |

## Winner: BTC (swin_tiny)

BTC with Cityscapes-pretrained Swin-Tiny backbone outperforms all other models:

- F1=0.5416 vs U-Net baseline of 0.4957 (+9% relative improvement)
- Converges in only 26 epochs (early stopping)
- Fast training (~36 seconds on RTX 4090)

## Key Findings

### Why FCSiam models fail with pretrained weights

FCSiamDiff and FCSiamConc use Siamese encoders that process both images independently.
With ImageNet pretrained encoders, both branches produce very similar initial feature maps
for any pair of images — the feature difference used by FCSiamDiff is near-zero,
and the concatenated features of FCSiamConc carry no change signal. At lr=0.0001,
the model cannot adapt fast enough to break this symmetry within 50 epochs.
Without pretrained weights (weights=False), FCSiamDiff converges to F1=0.4986, on par with U-Net.

### Why FCSiamConc consistently fails

FCSiamConc collapses to predicting all-negative in every configuration tried.
This may be a task/architecture mismatch — the concatenation approach may require
longer training, different LR, or more data to learn discriminative features.

### Why ChangeViT fails

ChangeViT with ViT-small backbone converges too quickly to a trivial solution.
With patch_size=256, the training set yields very few patches per epoch; the pretrained
ViT features dominate but don't transfer well to binary change detection in 50 epochs.

## Tutorial Choice

- **Primary model**: BTC (swin_tiny, weights=True) — best F1, modern Swin+UPerNet architecture
- **Comparison model**: FCSiamDiff (resnet18, weights=False) — classic Siamese baseline
  - Both are CD-specific architectures (neither is repurposed from segmentation)
  - Architecturally distinct: BTC uses Swin+UPerNet, FCSiamDiff uses ResNet Siamese diff

## Checkpoints

- `btc` (weights=True): `/tmp/cd_benchmark/btc/`
- `fcsiamdiff` (weights=False): `/tmp/cd_benchmark/fcsiamdiff_no_weights/`
