# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Run the EuroSAT SSL sweep across multiple GPUs.

Each config is a single-GPU run, so the sweep is embarrassingly parallel: one job
per GPU, with jobs pulled off a queue as GPUs free up.

This machine caps the whole session at a cgroup memory limit (96 GiB), which is
shared by every job and its dataloader workers. The scheduler therefore checks
current cgroup usage before launching and waits if the headroom is too small,
rather than letting the kernel OOM-kill a run several hours in.

Usage:
    python scripts/run_sweep.py --configs 'configs/resnet50/*.yaml'
    python scripts/run_sweep.py --configs 'configs/**/*.yaml' --gpus 0,1,2,3
"""

import argparse
import csv
import glob
import os
import pathlib
import subprocess
import sys
import time

import yaml

CGROUP_CURRENT = pathlib.Path('/sys/fs/cgroup/memory.current')
CGROUP_STAT = pathlib.Path('/sys/fs/cgroup/memory.stat')
CGROUP_MAX = pathlib.Path('/sys/fs/cgroup/memory.max')


def cgroup_usage() -> tuple[float, float]:
    """Read unreclaimable cgroup memory usage and the limit.

    ``memory.current`` counts page cache, which this workload fills by reading the
    dataset every epoch. That cache is reclaimed under pressure rather than
    causing an OOM kill, so scheduling on it would idle GPUs for no reason. Only
    anonymous memory and slab are counted here.

    Returns:
        Unreclaimable used memory and the limit, in GiB. Returns ``(0, inf)`` if
        unavailable.
    """
    try:
        raw = CGROUP_MAX.read_text().strip()
        total = float('inf') if raw == 'max' else int(raw) / 1024**3
        stats = {}
        for line in CGROUP_STAT.read_text().splitlines():
            key, _, value = line.partition(' ')
            stats[key] = int(value)
        used = (stats.get('anon', 0) + stats.get('slab', 0)) / 1024**3
    except (OSError, ValueError):
        return 0.0, float('inf')
    return used, total


def is_complete(run_dir: pathlib.Path, config: pathlib.Path) -> bool:
    """Check whether a run already trained to completion.

    The presence of ``last.ckpt`` is not sufficient: it is rewritten on every
    periodic checkpoint, so a run that is only part way through has one too. The
    logged epoch is compared against the config's ``max_epochs`` instead.

    Args:
        run_dir: Output directory for the run.
        config: Path to the run's config file.

    Returns:
        True if the run reached its final epoch.
    """
    if not (run_dir / 'checkpoints' / 'last.ckpt').exists():
        return False
    metrics = run_dir / 'csv' / 'metrics.csv'
    if not metrics.exists():
        return False
    try:
        with open(config) as f:
            max_epochs = int(yaml.safe_load(f)['trainer']['max_epochs'])
        last_epoch = -1
        with open(metrics) as f:
            for row in csv.DictReader(f):
                if row.get('epoch'):
                    last_epoch = max(last_epoch, int(row['epoch']))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return last_epoch >= max_epochs - 1


def main() -> None:
    """Schedule every config across the available GPUs."""
    here = pathlib.Path(__file__).resolve().parent.parent
    # This runs for hours behind a redirect; keep progress visible as it happens.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--configs',
        default='configs/**/*.yaml',
        help='comma-separated globs, run in the order given (highest priority first)',
    )
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument(
        '--headroom',
        type=float,
        default=10.0,
        help='GiB of cgroup memory to keep free before launching a new job',
    )
    parser.add_argument('--force', action='store_true', help='rerun finished runs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # Globs are expanded in the order given so that the runs most likely to
    # finish can be scheduled ahead of slower ones.
    configs: list[pathlib.Path] = []
    for pattern in args.configs.split(','):
        for path in sorted(glob.glob(pattern.strip(), recursive=True)):
            if pathlib.Path(path) not in configs:
                configs.append(pathlib.Path(path))
    if not configs:
        sys.exit(f'No configs matched {args.configs}')

    pending = []
    for config in configs:
        if is_complete(here / 'outputs' / config.stem, config) and not args.force:
            print(f'skip {config.stem} (already trained)')
            continue
        pending.append(config)

    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]
    print(f'{len(pending)} runs over {len(gpus)} GPUs')
    if args.dry_run:
        for config in pending:
            print(f'  would run {config}')
        return

    free_gpus = list(gpus)
    running: list[tuple[subprocess.Popen[bytes], str, str, float]] = []
    queue = list(pending)
    while queue or running:
        # Reap finished jobs.
        for entry in list(running):
            process, gpu, name, start = entry
            if process.poll() is not None:
                status = (
                    'ok' if process.returncode == 0 else f'FAIL {process.returncode}'
                )
                mins = (time.time() - start) / 60
                print(
                    f'[{time.strftime("%H:%M:%S")}] {name}: {status} ({mins:.1f} min)'
                )
                running.remove(entry)
                free_gpus.append(gpu)

        used, total = cgroup_usage()
        while queue and free_gpus:
            if total - used < args.headroom:
                print(
                    f'[{time.strftime("%H:%M:%S")}] waiting for memory: '
                    f'{used:.1f}/{total:.1f} GiB used'
                )
                break
            config = queue.pop(0)
            gpu = free_gpus.pop(0)
            run_dir = here / 'outputs' / config.stem
            run_dir.mkdir(parents=True, exist_ok=True)
            log = open(run_dir / 'train.log', 'w')
            command = [sys.executable, '-m', 'torchgeo', 'fit', '--config', str(config)]
            process = subprocess.Popen(
                command,
                cwd=here,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu},
            )
            print(f'[{time.strftime("%H:%M:%S")}] launch {config.stem} on GPU {gpu}')
            running.append((process, gpu, config.stem, time.time()))
            # Stagger launches so simultaneous dataset scans do not spike memory.
            time.sleep(20)
            used, total = cgroup_usage()

        time.sleep(15)

    print('sweep complete')


if __name__ == '__main__':
    main()
