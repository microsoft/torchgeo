# EuroSAT SSL benchmark

A reproducible, TorchGeo-native pretraining sweep for the SimCLR, MoCo v3, and
BYOL tasks on EuroSAT, plus a frozen-feature kNN probe for scoring the resulting
checkpoints.

Everything trains through `torchgeo fit` with stock `EuroSATDataModule` and stock
tasks — no custom datamodule, task, or augmentation — so the numbers reflect what
a user gets out of the box today.

## Protocol

Follows `docs/user/ssl_benchmark.rst`:

| Setting       | Value                                                                      |
| ------------- | -------------------------------------------------------------------------- |
| Dataset       | EuroSAT, all 13 Sentinel-2 bands, TorchGeo splits (16,200 / 5,400 / 5,400) |
| Normalization | `EuroSATDataModule` default (per-band `MEAN`/`STD`)                        |
| Input size    | 224x224, produced by each task's own `RandomResizedCrop` from native 64x64 |
| Pretraining   | 60 epochs, batch size 128, `16-mixed`, one GPU, seed 0                     |
| Features      | `forward_head(forward_features(x), pre_logits=True)`, unaugmented images   |
| Probe         | `KNeighborsClassifier(n_neighbors=5)`, Euclidean                           |
| Scaling       | fit on raw and on `StandardScaler` features, report the better             |
| Selection     | best learning rate on validation, then read test                           |

Every SSL task's `validation_step` is a no-op, so the configs set
`limit_val_batches: 0` and model selection happens externally in `eval_knn.py`.

## Layout

```
configs/{resnet50,vit_small}/<task>_<encoder>_lr<rate>.yaml   # committed
scripts/make_configs.py   # generates the configs
scripts/run_sweep.py      # schedules runs across GPUs, respects the cgroup RAM cap
scripts/eval_knn.py       # frozen-feature kNN probe over checkpoints
scripts/make_table.py     # aggregates results into a table
outputs/                  # checkpoints + logs (gitignored)
results/                  # per-run JSON scores
```

## Running

Requires a Python environment whose torch build matches the NVIDIA driver, with
`scikit-learn` available. On this machine that is
`/home/davrob/.conda/envs/torchgeo/bin/python` (torch 2.10+cu128); the default
`ai4gl-base` env has a cu130 build that cannot initialize CUDA against the 12.8
driver.

```bash
cd experiments/eurosat_benchmarking
ln -s /path/to/eurosat data/eurosat        # dataset root
PYTHON=/home/davrob/.conda/envs/torchgeo/bin/python ./run_all.sh
```

Or step by step:

```bash
python scripts/make_configs.py
python scripts/run_sweep.py --configs 'configs/resnet50/*.yaml' --gpus 0,1,2,3
python scripts/eval_knn.py --all
python scripts/make_table.py
```

`run_sweep.py` skips runs that already have a `last.ckpt`, so it is safe to
re-run after an interruption. `--configs` takes comma-separated globs that are
scheduled in the order given, which is how BYOL is deprioritized.

### Memory

This machine caps the session at a 96 GiB cgroup limit shared by every job and
its dataloader workers. The scheduler counts only anonymous + slab memory when
deciding whether to launch, because `memory.current` also counts page cache,
which this workload fills by re-reading the dataset each epoch and which the
kernel reclaims rather than OOM-killing. Configs use `num_workers: 3` so that
eight concurrent jobs fit within the cap.

## Status

All 24 runs completed 60 epochs and are fully scored: SimCLR and MoCo v3 in
~1.4–2 h each, BYOL in ~11 h each (its augmentation pipeline is ~7x slower per
step; see `FINDINGS.md`).

To re-run or extend:

```bash
# training skips runs that already reached max_epochs
python scripts/run_sweep.py --configs 'configs/**/*.yaml'
# scoring skips checkpoints that already have a result
python scripts/eval_knn.py --all --glob '*/checkpoints/*.ckpt'
python scripts/make_table.py
```

## Findings

`FINDINGS.md` records what this sweep says about the current state of TorchGeo
SSL training, including two failure modes worth knowing about: MoCo v3 diverges
to NaN or silently collapses over much of its learning rate range, and every SSL
task's `validation_step` is a no-op so nothing in-framework can detect it.
