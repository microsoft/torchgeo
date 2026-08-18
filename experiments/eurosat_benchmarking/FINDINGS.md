# Findings: the current state of TorchGeo SSL training

Notes from running the SimCLR, MoCo v3, and BYOL tasks on EuroSAT through
`torchgeo fit` with the stock `EuroSATDataModule`: 60 epochs, batch size 128,
13 bands, `16-mixed`, seed 0, one V100 per run. See `README.md` for the protocol
and `results/` for the raw scores.

Reference floors from `docs/user/ssl_benchmark.rst`: image statistics **0.8937**,
random init ResNet-50 **0.8622**, supervised ImageNet ResNet-50 **0.8948** /
ViT-S/16 **0.9178**.

## Headline

| Task    | Encoder   | Selected lr | val    | test       |
| ------- | --------- | ----------- | ------ | ---------- |
| MoCo v3 | ResNet-50 | 1e-2        | 0.9448 | **0.9494** |
| MoCo v3 | ViT-S/16  | 1e-4        | 0.9380 | 0.9413     |
| SimCLR  | ResNet-50 | 1.5         | 0.9326 | 0.9356     |
| SimCLR  | ViT-S/16  | 1.5         | 0.9330 | 0.9296     |
| BYOL    | ResNet-50 | 1e-3        | 0.9309 | 0.9294     |
| BYOL    | ViT-S/16  | 1e-4        | 0.9263 | 0.9241     |

All 24 runs completed 60 epochs. All three tasks work and beat every floor at
their best learning rate. MoCo v3 is the strongest, and also by far the most
fragile: it is the only task with runs that diverged or collapsed. BYOL is the
weakest of the three and, at ~11 h per run versus ~1.4–2 h, by far the slowest.

## Full sweep (final checkpoints, 60 epochs)

| run                  | lr            | val        | test       | emb std | mean pairwise cos   |
| -------------------- | ------------- | ---------- | ---------- | ------- | ------------------- |
| byol_resnet50        | 1e-4          | 0.9170     | 0.9211     | 0.0157  | 0.490               |
| byol_resnet50        | 3e-4          | 0.9272     | 0.9309     | 0.0162  | 0.459               |
| **byol_resnet50**    | **1e-3**      | **0.9309** | **0.9294** | 0.0163  | 0.428               |
| byol_resnet50        | 3e-3          | 0.8946     | 0.8941     | 0.0131  | 0.574               |
| **byol_vit_small**   | **1e-4**      | **0.9263** | **0.9241** | 0.0338  | 0.526               |
| byol_vit_small       | 3e-4          | 0.8815     | 0.8837     | 0.0400  | 0.326               |
| byol_vit_small       | 1e-3          | 0.8170     | 0.8146     | 0.0307  | 0.560               |
| byol_vit_small       | 3e-3          | 0.8272     | 0.8372     | 0.0131  | 0.926               |
| moco_resnet50        | 1e-4          | 0.9183     | 0.9243     | 0.0137  | 0.575               |
| moco_resnet50        | 1e-3          | 0.9398     | 0.9422     | 0.0160  | 0.399               |
| **moco_resnet50**    | **1e-2**      | **0.9448** | **0.9494** | 0.0142  | 0.428               |
| moco_resnet50        | 1e-1          | —          | —          | —       | diverged (NaN)      |
| **moco_vit_small**   | **1e-4**      | **0.9380** | **0.9413** | 0.0262  | 0.722               |
| moco_vit_small       | 1e-3          | 0.9298     | 0.9426     | 0.0104  | 0.955               |
| moco_vit_small       | 1e-2          | —          | —          | —       | collapsed, then NaN |
| moco_vit_small       | 1e-1          | —          | —          | —       | diverged (NaN)      |
| simclr_resnet50      | 0.15          | 0.9150     | 0.9152     | 0.0121  | 0.677               |
| simclr_resnet50      | 0.5           | 0.9307     | 0.9330     | 0.0141  | 0.554               |
| **simclr_resnet50**  | **1.5**       | **0.9326** | **0.9356** | 0.0133  | 0.592               |
| simclr_resnet50      | 4.8 (default) | 0.9280     | 0.9317     | 0.0123  | 0.607               |
| simclr_vit_small     | 0.15          | 0.9185     | 0.9180     | 0.0348  | 0.508               |
| simclr_vit_small     | 0.5           | 0.9215     | 0.9269     | 0.0166  | 0.883               |
| **simclr_vit_small** | **1.5**       | **0.9330** | **0.9296** | 0.0328  | 0.549               |
| simclr_vit_small     | 4.8 (default) | 0.9230     | 0.9276     | 0.0323  | 0.552               |

Bold rows are the learning rate selected on validation accuracy for that task and
encoder.

## 1. The previously published numbers reproduce

The rows removed from `docs/user/ssl_benchmark.rst` pending reproduction line up
with this sweep, at the same selected learning rates:

| Row                | Previously reported | This sweep | lr                         |
| ------------------ | ------------------- | ---------- | -------------------------- |
| MoCo v3, ResNet-50 | 0.9476              | 0.9494     | 1e-2 (both)                |
| MoCo v3, ViT-S/16  | 0.9396              | 0.9413     | 1e-3 previously, 1e-4 here |
| SimCLR, ResNet-50  | 0.9393              | 0.9356     | 1.5 (both)                 |
| SimCLR, ViT-S/16   | 0.9326              | 0.9296     | 1.5 (both)                 |

Three of four are within ~0.002 and SimCLR/ResNet-50 is within 0.004. The MoCo
ViT row is the one real difference: the previously reported 1e-3 optimum is not
reproducible here because at 1e-3 the ViT run ends up near-collapsed
(`emb std` 0.0104, cosine 0.955) and 1e-2 collapses outright, so the sweep
selects 1e-4 instead.

## 2. MoCo v3 diverges to NaN, sometimes silently and late

Three of the eight MoCo runs failed: two diverged to NaN outright, and one
collapsed progressively before diverging:

| moco_vit_small lr 1e-2 | val    | test   | emb std    |
| ---------------------- | ------ | ------ | ---------- |
| epoch 19               | 0.8294 | 0.8317 | 0.0080     |
| epoch 39               | 0.7596 | 0.7528 | **0.0005** |
| epoch 59               | NaN    | NaN    | —          |

Accuracy fell _below the random-init floor_ and the embedding standard deviation
reached 0.0005 — a fully collapsed representation — while the training loss
stayed essentially flat (9.66 at epoch 14, 9.86 at epoch 29). This is exactly the
failure mode the benchmark page warns about: **the loss curve gives no
indication.** Nothing in the task reports it, and `torchgeo fit` exits 0.

The NaN runs also complete "successfully": `moco_resnet50` at lr 1e-1 trained all
60 epochs with a NaN loss and wrote a checkpoint whose features are entirely NaN.
The failure only surfaces downstream, when the probe raises
`ValueError: Input X contains NaN`.

This matters because MoCo's default is `lr=9.6` — the linear scaling rule at
batch 4096 (`0.6 * 4096 / 256`). At batch 128 the scaled rate would be 0.3, and
1e-1 already diverges. **A user who accepts the default learning rate gets a
silently NaN model**, with no warning from the task.

SimCLR's default of 4.8 is also a batch-4096 rate, but SimCLR uses LARS, whose
layer-wise trust ratio makes it far more forgiving: 4.8 still scores 0.9317, only
0.004 below the best rate. Every SimCLR run trained stably.

## 3. BYOL is ~7x slower per step than SimCLR or MoCo

Measured on one V100, batch 64, 13 channels, 64x64 input:

| Augmentation pipeline          | ms / batch |
| ------------------------------ | ---------- |
| BYOL `SimCLRAugmentation(224)` | **1621**   |
| MoCo `moco_augmentations`      | 218        |
| SimCLR `simclr_augmentations`  | 238        |

The cause is the ordering inside `SimCLRAugmentation` (`torchgeo/tasks/byol.py`):
it runs `K.Resize(224)` **first**, then applies `RandomHorizontalFlip`,
`RandomGaussianBlur`, and `RandomResizedCrop` to the full 224x224x13 tensor.
SimCLR and MoCo instead `RandomResizedCrop` straight from the native 64x64 to
224x224, so their random ops run on the small tensor and only the final resize
touches the big one.

BYOL calls `augment` twice per step (once per view), so a step spends ~3.2 s in
augmentation alone. In the sweep this is ~8.4 s/step versus ~1.2 s/step for MoCo,
with GPU utilization of **4–8%** versus 90%+: the GPU idles while Kornia
dispatches many small kernels over an oversized tensor.

Consequence: 60 epochs of BYOL took **~11 h per run** (581–695 min), versus ~1.4 h
for SimCLR and ~2 h for MoCo v3 on identical hardware — a 5–8x wall-clock penalty
for the weakest result of the three. BYOL runs are checkpointed every 2 epochs so
partial runs stay evaluable.

BYOL itself trains fine — it is just slow, and it needs a small learning rate.
Learning is smooth and monotonic on ResNet-50:

| byol_resnet50 lr 3e-4 | test   |
| --------------------- | ------ |
| epoch 1               | 0.8893 |
| epoch 5               | 0.9070 |
| epoch 9               | 0.9181 |
| epoch 15              | 0.9278 |
| epoch 59              | 0.9309 |

Most of BYOL's gain arrives in the first ~15 epochs; the remaining 45 epochs add
about 0.003. On ViT-S/16 it is much more rate-sensitive: 1e-4 reaches 0.9241 but
3e-3 ends at 0.8372 with a mean pairwise cosine of 0.926, i.e. close to collapse.

## 4. Every SSL task's `validation_step` is a no-op

`SimCLR`, `MoCo`, and `BYOL` all define `validation_step` as a no-op, so there is
no in-training signal — no validation metric, and nothing for `ModelCheckpoint`
to monitor. BYOL's `ReduceLROnPlateau` therefore monitors `train_loss`.

This is why the configs set `limit_val_batches: 0` and why model selection has to
happen out of band in `eval_knn.py`. Combined with section 2, a user has no
in-framework way to notice that a run has collapsed.

## 5. MoCo v3's warmup is hardcoded to 40 epochs

`torchgeo/tasks/moco.py` hardcodes `warmup_epochs = 40` for v3 regardless of
`max_epochs`. At 60 epochs, two thirds of training is linear warmup and only the
last third follows the cosine schedule. MoCo still wins the benchmark, but its
schedule is not what a 60-epoch run would normally imply, and the hardcoded value
is plausibly part of why its usable learning rate range is so narrow.

## 6. Environment note

The `ai4gl-base` conda environment ships torch 2.13.0+cu130, which cannot
initialize CUDA against this machine's 12.8 driver
(`RuntimeError: The NVIDIA driver on your system is too old`). All runs here use
`/home/davrob/.conda/envs/torchgeo` (torch 2.10.0+cu128), which is an editable
install of this repository.
