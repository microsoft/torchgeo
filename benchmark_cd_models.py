#!/usr/bin/env python3
"""Benchmark change-detection-specific models on OSCD100.

Runs fcsiamdiff, fcsiamconc, btc, and changevit with shared hyperparameters
and prints a markdown comparison table. Results saved to BENCHMARK_RESULTS.md.

Usage:
    python benchmark_cd_models.py
"""

import os
import tempfile
import time

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torchgeo.datamodules import OSCD100DataModule
from torchgeo.datasets import OSCD100
from torchgeo.trainers import ChangeDetectionTask

# Shared config
LR = 0.0001
LOSS = 'bce'
IN_CHANNELS = 3  # RGB: B04, B03, B02
MAX_EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 4

# Models to benchmark: (model_key, backbone, patch_size, batch_size)
# ChangeViT requires patch_size==img_size (default 256), smaller batch to fit VRAM
CONFIGS = [
    ('fcsiamdiff', 'resnet18', 64, 8),
    ('fcsiamconc', 'resnet18', 64, 8),
    ('btc', 'swin_tiny', 64, 8),
    ('changevit', 'vit_small_patch16_224', 256, 4),
]


def run_model(
    model_name: str,
    backbone: str,
    patch_size: int,
    batch_size: int,
    data_root: str,
    output_root: str,
) -> dict[str, object]:
    """Train and test a single model configuration."""
    print(f'\n{"=" * 70}')
    print(
        f'Model: {model_name}  backbone: {backbone}  patch: {patch_size}  bs: {batch_size}'
    )
    print('=' * 70)

    model_dir = os.path.join(output_root, model_name)
    os.makedirs(model_dir, exist_ok=True)

    datamodule = OSCD100DataModule(
        root=data_root,
        bands=OSCD100.rgb_bands,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=NUM_WORKERS,
        download=True,
    )

    task = ChangeDetectionTask(
        model=model_name,
        backbone=backbone,
        weights=True,
        loss=LOSS,
        in_channels=IN_CHANNELS,
        lr=LR,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss', dirpath=model_dir, save_top_k=1, save_last=False
    )
    early_stopping_cb = EarlyStopping(
        monitor='val_loss', min_delta=0.0, patience=PATIENCE
    )
    logger = TensorBoardLogger(save_dir=model_dir, name='logs')

    trainer = Trainer(
        callbacks=[checkpoint_cb, early_stopping_cb],
        log_every_n_steps=1,
        logger=logger,
        min_epochs=1,
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        enable_progress_bar=True,
    )

    t0 = time.time()
    trainer.fit(model=task, datamodule=datamodule)
    train_time = time.time() - t0

    results = trainer.test(model=task, datamodule=datamodule, ckpt_path='best')
    metrics = results[0]

    return {
        'model': model_name,
        'backbone': backbone,
        'f1': metrics.get('test_BinaryF1Score', float('nan')),
        'iou': metrics.get('test_BinaryJaccardIndex', float('nan')),
        'acc': metrics.get('test_BinaryAccuracy', float('nan')),
        'epochs': trainer.current_epoch,
        'time_min': train_time / 60,
        'ckpt': checkpoint_cb.best_model_path,
    }


def print_table(rows: list[dict[str, object]]) -> None:
    """Print benchmark results as a formatted table."""
    header = f'{"Model":<14} {"Backbone":<26} {"F1":>6} {"IoU":>6} {"Acc":>6} {"Epochs":>6} {"Time(m)":>8}'
    sep = '-' * len(header)
    print('\n' + sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f'{r["model"]:<14} {r["backbone"]:<26} '
            f'{r["f1"]:>6.4f} {r["iou"]:>6.4f} {r["acc"]:>6.4f} '
            f'{r["epochs"]:>6d} {r["time_min"]:>8.1f}'
        )
    print(sep)


def save_markdown(rows: list[dict[str, object]], path: str) -> None:
    """Write benchmark results to a markdown file."""
    lines = [
        '# CD Model Benchmark Results on OSCD100',
        '',
        f'Config: lr={LR}, loss={LOSS}, in_channels={IN_CHANNELS}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}',
        '',
        '| Model | Backbone | F1 | IoU | Accuracy | Epochs | Time (min) |',
        '|-------|----------|-----|-----|----------|--------|------------|',
    ]
    for r in rows:
        lines.append(
            f'| {r["model"]} | {r["backbone"]} | '
            f'{r["f1"]:.4f} | {r["iou"]:.4f} | {r["acc"]:.4f} | '
            f'{r["epochs"]} | {r["time_min"]:.1f} |'
        )
    lines += ['', '## Checkpoints']
    for r in rows:
        lines.append(f'- `{r["model"]}`: `{r["ckpt"]}`')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nResults saved to {path}')


def main() -> None:
    """Run benchmark across all model configurations."""
    torch.set_float32_matmul_precision('medium')

    data_root = os.path.join(tempfile.gettempdir(), 'oscd100')
    output_root = os.path.join(tempfile.gettempdir(), 'cd_benchmark')

    print(f'Data root:   {data_root}')
    print(f'Output root: {output_root}')

    rows = []
    for model_name, backbone, patch_size, batch_size in CONFIGS:
        try:
            result = run_model(
                model_name, backbone, patch_size, batch_size, data_root, output_root
            )
            rows.append(result)
        except Exception as e:
            print(f'ERROR running {model_name}: {e}')
            rows.append(
                {
                    'model': model_name,
                    'backbone': backbone,
                    'f1': float('nan'),
                    'iou': float('nan'),
                    'acc': float('nan'),
                    'epochs': 0,
                    'time_min': 0.0,
                    'ckpt': 'FAILED',
                }
            )

    print('\n\nFINAL RESULTS')
    print_table(rows)

    # Rank by F1
    valid = [r for r in rows if not (r['f1'] != r['f1'])]  # filter NaN
    if valid:
        best = max(valid, key=lambda r: r['f1'])
        print(f'\nBest model by F1: {best["model"]} (F1={best["f1"]:.4f})')

    save_markdown(rows, 'BENCHMARK_RESULTS.md')


if __name__ == '__main__':
    main()
