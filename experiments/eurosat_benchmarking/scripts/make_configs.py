# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Generate TorchGeo CLI configs for the EuroSAT SSL benchmark.

Each config pretrains one SSL task on EuroSAT without labels, following the
protocol in ``docs/user/ssl_benchmark.rst``: 13 Sentinel-2 bands, 224x224 inputs
produced by each task's own ``RandomResizedCrop``, batch size 128, 60 epochs,
mixed precision, one GPU, seed 0.

The learning rate is the only thing that varies within a task. The library
defaults follow the linear scaling rule at batch 4096 (MoCo v3 ``9.6`` is
``0.6 * 4096 / 256``, SimCLR ``4.8`` is ``0.3 * 4096 / 256``), which is far too
large at batch 128, so each task is swept over four rates spanning the plausible
range including its library default.

Usage:
    python scripts/make_configs.py [--data-root PATH] [--output-root PATH]

All paths in the generated configs are relative, so the configs are portable and
should be run with the experiment directory as the working directory.
"""

import argparse
import pathlib

import yaml


class _IndentedDumper(yaml.SafeDumper):
    """YAML dumper that indents list items under their parent key.

    Matches the style prettier enforces in CI, so regenerating the configs does
    not produce a formatting diff.
    """

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        """Always indent block sequences.

        Args:
            flow: Whether the collection is in flow style.
            indentless: Ignored; block sequences are always indented.
        """
        return super().increase_indent(flow, False)


# Protocol constants. Changing any of these makes results incomparable.
SEED = 0
MAX_EPOCHS = 60
BATCH_SIZE = 128
IN_CHANNELS = 13
SIZE = 224
PRECISION = '16-mixed'
NUM_WORKERS = 3

# Learning rate grids. The library default for each task is included so that the
# out-of-the-box behaviour is measured alongside the tuned behaviour.
LR_GRIDS: dict[str, list[float]] = {
    # LARS optimizer. 0.15 is the linear scaling rule at batch 128, 4.8 is the
    # library default (tuned for batch 4096).
    'simclr': [0.15, 0.5, 1.5, 4.8],
    # AdamW. The library default of 9.6 is a LARS-scale rate and diverges under
    # AdamW, so the grid spans the usable AdamW range instead.
    'moco': [1e-4, 1e-3, 1e-2, 1e-1],
    # AdamW, library default 1e-3.
    'byol': [1e-4, 3e-4, 1e-3, 3e-3],
}

ENCODERS: dict[str, str] = {
    'resnet50': 'resnet50',
    'vit_small': 'vit_small_patch16_224',
}


def model_block(task: str, encoder: str, lr: float) -> dict[str, object]:
    """Build the ``model`` section of the config.

    Args:
        task: One of ``simclr``, ``moco``, or ``byol``.
        encoder: timm model name.
        lr: Learning rate.

    Returns:
        The ``model`` config section.

    Raises:
        ValueError: If the task is unknown.
    """
    match task:
        case 'simclr':
            return {
                'class_path': 'SimCLR',
                'init_args': {
                    'model': encoder,
                    'in_channels': IN_CHANNELS,
                    'version': 2,
                    'lr': lr,
                    'size': SIZE,
                    # The memory bank is a SimCLR v2 distillation feature and is
                    # not part of this protocol.
                    'memory_bank_size': 0,
                },
            }
        case 'moco':
            return {
                'class_path': 'MoCo',
                'init_args': {
                    'model': encoder,
                    'in_channels': IN_CHANNELS,
                    'version': 3,
                    'lr': lr,
                    'size': SIZE,
                },
            }
        case 'byol':
            # BYOL takes no size argument; it resizes to 224x224 internally.
            return {
                'class_path': 'BYOL',
                'init_args': {'model': encoder, 'in_channels': IN_CHANNELS, 'lr': lr},
            }
        case _:
            raise ValueError(f'Unknown task: {task}')


def make_config(
    task: str, encoder_key: str, lr: float, data_root: str, run_dir: str
) -> dict[str, object]:
    """Build a full TorchGeo CLI config for one run.

    Args:
        task: One of ``simclr``, ``moco``, or ``byol``.
        encoder_key: Short encoder name used in the run name.
        lr: Learning rate.
        data_root: Path to the EuroSAT dataset.
        run_dir: Directory to write checkpoints and logs to.

    Returns:
        The config as a dictionary.
    """
    # BYOL's augmentation pipeline is roughly 7x slower per step than SimCLR's or
    # MoCo's, so a BYOL run may not reach 60 epochs in a session. It is
    # checkpointed far more often so that partial runs are still evaluable.
    checkpoint_every = 2 if task == 'byol' else 20
    return {
        'seed_everything': SEED,
        'trainer': {
            'accelerator': 'gpu',
            'devices': 1,
            'max_epochs': MAX_EPOCHS,
            'precision': PRECISION,
            'benchmark': True,
            'log_every_n_steps': 10,
            'enable_progress_bar': False,
            # Every SSL task's validation_step is a no-op, so model selection
            # happens externally via the kNN probe in eval_knn.py.
            'limit_val_batches': 0,
            'num_sanity_val_steps': 0,
            'default_root_dir': run_dir,
            'callbacks': [
                {
                    'class_path': 'lightning.pytorch.callbacks.ModelCheckpoint',
                    'init_args': {
                        'dirpath': f'{run_dir}/checkpoints',
                        'filename': 'epoch{epoch:03d}',
                        'auto_insert_metric_name': False,
                        'every_n_epochs': checkpoint_every,
                        'save_top_k': -1,
                        'save_last': True,
                    },
                },
                {
                    'class_path': 'lightning.pytorch.callbacks.LearningRateMonitor',
                    'init_args': {'logging_interval': 'epoch'},
                },
            ],
            'logger': [
                {
                    'class_path': 'lightning.pytorch.loggers.CSVLogger',
                    'init_args': {
                        'save_dir': run_dir,
                        'name': 'csv',
                        'version': '',
                        'flush_logs_every_n_steps': 50,
                    },
                }
            ],
        },
        'model': model_block(task, ENCODERS[encoder_key], lr),
        'data': {
            'class_path': 'EuroSATDataModule',
            'init_args': {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS},
            'dict_kwargs': {'root': data_root, 'download': False},
        },
    }


def lr_tag(lr: float) -> str:
    """Format a learning rate as a filename-safe tag.

    Args:
        lr: Learning rate.

    Returns:
        A filename-safe string, e.g. ``lr0p01`` or ``lr4p8``.
    """
    return 'lr' + f'{lr:g}'.replace('.', 'p').replace('-', 'm')


def main() -> None:
    """Generate every config in the sweep."""
    here = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/eurosat')
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--config-dir', default=str(here / 'configs'))
    args = parser.parse_args()

    config_dir = pathlib.Path(args.config_dir)
    count = 0
    for encoder_key in ENCODERS:
        for task, lrs in LR_GRIDS.items():
            for lr in lrs:
                name = f'{task}_{encoder_key}_{lr_tag(lr)}'
                run_dir = f'{args.output_root}/{name}'
                config = make_config(task, encoder_key, lr, args.data_root, run_dir)
                path = config_dir / encoder_key / f'{name}.yaml'
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    yaml.dump(
                        config,
                        f,
                        Dumper=_IndentedDumper,
                        sort_keys=False,
                        default_flow_style=False,
                    )
                count += 1
    print(f'Wrote {count} configs to {config_dir}')


if __name__ == '__main__':
    main()
