# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Aggregate kNN evaluation results into a summary table.

Reads the per-run JSON records written by ``eval_knn.py``, selects the best
learning rate for each task and encoder on validation accuracy, and prints both
the full sweep and the selected rows as reStructuredText suitable for
``docs/user/ssl_benchmark.rst``.

Usage:
    python scripts/make_table.py
"""

import argparse
import csv
import json
import pathlib
from typing import Any

TASK_LABEL = {'simclr': 'SimCLR', 'moco': 'MoCo v3', 'byol': 'BYOL'}
ENCODER_LABEL = {'resnet50': 'ResNet-50', 'vit': 'ViT-S/16'}


def parse_run(name: str) -> tuple[str, str]:
    """Split a run directory name into task and encoder.

    Args:
        name: Run name, e.g. ``simclr_resnet50_lr1p5``.

    Returns:
        The task key and encoder key.
    """
    task, _, rest = name.partition('_')
    encoder = 'resnet50' if rest.startswith('resnet50') else 'vit'
    return task, encoder


def load_records(results: pathlib.Path) -> list[dict[str, Any]]:
    """Load every evaluation record.

    Args:
        results: Directory of JSON records.

    Returns:
        Records sorted by task, encoder, and learning rate.
    """
    records = []
    for path in sorted(results.glob('*.json')):
        with open(path) as f:
            record = json.load(f)
        task, encoder = parse_run(record['run'])
        record['task'], record['encoder'] = task, encoder
        records.append(record)
    return sorted(records, key=lambda r: (r['task'], r['encoder'], r.get('lr') or 0.0))


def main() -> None:
    """Print the full sweep and the per-task best rows."""
    here = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=pathlib.Path, default=here / 'results')
    parser.add_argument('--csv', type=pathlib.Path, default=here / 'results.csv')
    args = parser.parse_args()

    records = load_records(args.results)
    ok = [r for r in records if 'error' not in r]
    failed = [r for r in records if 'error' in r]
    # Only fully trained checkpoints are comparable for learning rate selection.
    final = [r for r in ok if r.get('checkpoint_name', 'last') == 'last']

    print(f'{len(ok)} evaluated, {len(failed)} failed\n')
    header = (
        f'{"run":34} {"lr":>8} {"val":>7} {"test":>7} {"emb std":>8} {"cos sim":>8}'
    )
    print(header)
    print('-' * len(header))
    for r in ok:
        print(
            f'{r["run"]:34} {r["lr"]:>8.4g} {r["val_acc"]:>7.4f} '
            f'{r["test_acc"]:>7.4f} {r["embedding_std"]:>8.4f} '
            f'{r["mean_pairwise_cosine"]:>8.4f}'
        )
    for r in failed:
        print(f'{r["run"]:34} {"FAILED":>8} {r["error"][:60]}')

    if ok:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(ok[0].keys()))
            writer.writeheader()
            writer.writerows(ok)
        print(f'\nwrote {args.csv}')

    # Best learning rate per task and encoder, selected on validation accuracy.
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in final:
        key = (r['task'], r['encoder'])
        if key not in best or r['val_acc'] > best[key]['val_acc']:
            best[key] = r

    print('\nSelected on validation accuracy:\n')
    print('   * - Task\n     - Encoder\n     - Test acc\n     - lr')
    for (task, encoder), r in sorted(best.items()):
        print(
            f'   * - {TASK_LABEL.get(task, task)}\n'
            f'     - {ENCODER_LABEL[encoder]}\n'
            f'     - {r["test_acc"]:.4f}\n'
            f'     - {r["lr"]:g}'
        )


if __name__ == '__main__':
    main()
