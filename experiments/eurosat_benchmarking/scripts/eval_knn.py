# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Evaluate SSL checkpoints on EuroSAT with a frozen-feature kNN probe.

Implements the scoring half of the protocol in ``docs/user/ssl_benchmark.rst``:
freeze the encoder, extract ``forward_head(forward_features(x), pre_logits=True)``
on unaugmented images, and fit a k-nearest-neighbour classifier on the frozen
features. The probe is fit on raw features and on ``StandardScaler`` features and
the better of the two is reported, as the reference paper does.

The learning rate is selected on validation accuracy; test accuracy is reported
alongside so the selection can be audited, but should only be read once.

Training normalizes with the EuroSAT per-band mean and standard deviation inside
``on_after_batch_transfer`` and then upsamples 64x64 to 224x224 inside the task's
own ``RandomResizedCrop``. That ordering (normalize, then resize) is reproduced
here, minus the random augmentation.

Usage:
    python scripts/eval_knn.py --checkpoint outputs/<run>/checkpoints/last.ckpt
    python scripts/eval_knn.py --all  # every run under outputs/
"""

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from torchgeo.datamodules import EuroSATDataModule

# Protocol constants.
IMAGE_SIZE = 224
N_NEIGHBORS = 5


def infer_backbone_prefix(state_dict: dict[str, Tensor]) -> str:
    """Find the state dict prefix holding the timm encoder.

    SimCLR and MoCo store the encoder as ``backbone``; BYOL nests it inside its
    ``BackboneWrapper`` as ``model.backbone.model``. MoCo additionally holds a
    ``backbone_momentum`` copy, which is deliberately not matched here.

    Args:
        state_dict: Checkpoint state dict.

    Returns:
        The prefix of the encoder weights, including the trailing dot.

    Raises:
        ValueError: If no known encoder prefix is present.
    """
    for prefix in ('model.backbone.model.', 'backbone.'):
        if any(k.startswith(prefix) for k in state_dict):
            return prefix
    raise ValueError('Could not locate a timm encoder in the checkpoint')


def load_backbone(checkpoint: pathlib.Path) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the frozen timm encoder from a checkpoint.

    The task class is not instantiated: only the encoder is needed, and rebuilding
    it directly avoids constructing projection heads and momentum encoders.

    Args:
        checkpoint: Path to a Lightning checkpoint.

    Returns:
        The encoder in eval mode, and the checkpoint hyperparameters.
    """
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = dict(ckpt['hyper_parameters'])
    state_dict = ckpt['state_dict']
    prefix = infer_backbone_prefix(state_dict)
    weights = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    backbone = timm.create_model(
        hparams['model'], in_chans=hparams['in_channels'], num_classes=0
    )
    # strict=True: a silent mismatch here would evaluate a partly random encoder.
    backbone.load_state_dict(weights, strict=True)
    return backbone.eval(), hparams


@torch.no_grad()
def extract_features(
    backbone: nn.Module, datamodule: EuroSATDataModule, split: str, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frozen features for one split.

    Args:
        backbone: Frozen timm encoder.
        datamodule: EuroSAT datamodule, already set up.
        split: One of ``train``, ``val``, or ``test``.
        device: Device to run on.

    Returns:
        Features of shape ``(N, D)`` and integer labels of shape ``(N,)``.
    """
    loader = getattr(datamodule, f'{split}_dataloader')()
    features, labels = [], []
    for batch in loader:
        x = batch['image'].to(device, non_blocking=True)
        # Reproduce the training preprocessing: per-band standardization from the
        # datamodule, then upsample to the size the encoder was trained at.
        x = datamodule.aug({'image': x})['image']
        x = F.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False
        )
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            z = backbone.forward_head(backbone.forward_features(x), pre_logits=True)
        features.append(z.float().flatten(1).cpu())
        labels.append(batch['label'])
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def collapse_diagnostics(features: np.ndarray, sample: int = 2048) -> dict[str, float]:
    """Measure how degenerate a representation is.

    A collapsed encoder maps every image to nearly the same vector, which shows up
    as a near-zero standard deviation across L2-normalized embeddings and a mean
    pairwise cosine similarity near one.

    Args:
        features: Features of shape ``(N, D)``.
        sample: Number of rows to subsample for the pairwise similarity.

    Returns:
        Embedding standard deviation and mean pairwise cosine similarity.
    """
    x = torch.from_numpy(features).float()
    normalized = F.normalize(x, dim=1)
    std = float(normalized.std(dim=0).mean())

    rng = np.random.default_rng(0)
    idx = rng.choice(len(normalized), size=min(sample, len(normalized)), replace=False)
    subset = normalized[idx]
    similarity = subset @ subset.T
    n = len(subset)
    # Exclude the diagonal, which is always 1.
    off_diagonal = (similarity.sum() - n) / (n * (n - 1))
    return {'embedding_std': std, 'mean_pairwise_cosine': float(off_diagonal)}


def knn_accuracy(
    train: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, eval_y: np.ndarray
) -> dict[str, float]:
    """Score frozen features with a kNN probe, raw and standardized.

    Args:
        train: Training features.
        train_y: Training labels.
        eval_x: Evaluation features.
        eval_y: Evaluation labels.

    Returns:
        Accuracy on raw features, on standardized features, and the better one.
    """
    scores = {}
    for name in ('raw', 'scaled'):
        if name == 'raw':
            a, b = train, eval_x
        else:
            scaler = StandardScaler()
            a, b = scaler.fit_transform(train), scaler.transform(eval_x)
        probe = KNeighborsClassifier(n_neighbors=N_NEIGHBORS).fit(a, train_y)
        scores[name] = float(probe.score(b, eval_y))
    scores['best'] = max(scores['raw'], scores['scaled'])
    return scores


def evaluate(
    checkpoint: pathlib.Path,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a single checkpoint.

    Args:
        checkpoint: Path to a Lightning checkpoint.
        data_root: Path to the EuroSAT dataset.
        batch_size: Feature extraction batch size.
        num_workers: Dataloader workers.
        device: Device to run on.

    Returns:
        A result record with val and test accuracy plus collapse diagnostics.
    """
    start = time.time()
    backbone, hparams = load_backbone(checkpoint)
    backbone = backbone.to(device)

    datamodule = EuroSATDataModule(
        batch_size=batch_size, num_workers=num_workers, root=data_root, download=False
    )
    datamodule.setup('fit')
    datamodule.setup('test')
    datamodule.aug = datamodule.aug.to(device)

    train_x, train_y = extract_features(backbone, datamodule, 'train', device)
    val_x, val_y = extract_features(backbone, datamodule, 'val', device)
    test_x, test_y = extract_features(backbone, datamodule, 'test', device)

    val = knn_accuracy(train_x, train_y, val_x, val_y)
    test = knn_accuracy(train_x, train_y, test_x, test_y)

    return {
        'checkpoint': str(checkpoint),
        'run': checkpoint.parent.parent.name,
        'model': hparams.get('model'),
        'lr': hparams.get('lr'),
        'in_channels': hparams.get('in_channels'),
        'feature_dim': int(train_x.shape[1]),
        'val_acc': val['best'],
        'val_acc_raw': val['raw'],
        'val_acc_scaled': val['scaled'],
        'test_acc': test['best'],
        'test_acc_raw': test['raw'],
        'test_acc_scaled': test['scaled'],
        **collapse_diagnostics(train_x),
        'eval_seconds': round(time.time() - start, 1),
    }


def main() -> None:
    """Evaluate one checkpoint or every checkpoint under the outputs directory."""
    here = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=pathlib.Path)
    parser.add_argument('--all', action='store_true')
    parser.add_argument(
        '--glob',
        default='*/checkpoints/last.ckpt',
        help='checkpoint glob used by --all, relative to --outputs',
    )
    parser.add_argument('--outputs', type=pathlib.Path, default=here / 'outputs')
    parser.add_argument('--data-root', default=str(here / 'data' / 'eurosat'))
    parser.add_argument('--results', type=pathlib.Path, default=here / 'results')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.all:
        checkpoints = sorted(args.outputs.glob(args.glob))
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        parser.error('pass --checkpoint or --all')

    args.results.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    for checkpoint in checkpoints:
        run = checkpoint.parent.parent.name
        out = args.results / f'{run}__{checkpoint.stem}.json'
        if out.exists():
            print(f'skip {out.name} (already evaluated)')
            continue
        try:
            record = evaluate(
                checkpoint, args.data_root, args.batch_size, args.num_workers, device
            )
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the sweep
            record = {
                'checkpoint': str(checkpoint),
                'run': run,
                'error': f'{type(e).__name__}: {e}',
            }
        record['checkpoint_name'] = checkpoint.stem
        with open(out, 'w') as f:
            json.dump(record, f, indent=2)
        print(json.dumps(record))


if __name__ == '__main__':
    main()
