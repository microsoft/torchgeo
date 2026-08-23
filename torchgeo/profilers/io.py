# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
"""A custom I/O Profiler."""

import time
import warnings
from collections import defaultdict
from typing import override

from lightning.pytorch.profilers import Profiler


class IOProfiler(Profiler):
    """A custom I/O Benchmarking Profiler.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dirpath: str | None = None,
        filename: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialise profiler.

        .. deprecated:: 0.11
            The *batch_size* parameter.

        Args:
            dirpath: root directory to save profiler's results
            filename: name of the file where the profiler's results will be saved
            batch_size: batch size used by data loader
        """
        super().__init__(dirpath=dirpath, filename=filename)
        self.start_time = {}
        self.action_count = defaultdict(int)
        self.end_time = {}
        self.action_total_time = defaultdict(float)
        self.info = {}
        if batch_size is not None:
            warnings.warn('The batch_size parameter is deprecated.', DeprecationWarning)

    @override
    def start(self, action_name: str) -> None:
        """Start recording.

        Args:
            action_name: name of the action that should be profiled
        """
        self.start_time[action_name] = time.perf_counter()
        split = (
            'train'
            if 'train_dataloader_next' in action_name
            else 'val'
            if 'val_next' in action_name
            else None
        )

        if split is not None and split not in self.info:
            loader = getattr(
                self._lightning_module.trainer,
                f'{split}_{"dataloaders" if split == "val" else "dataloader"}',
            )
            self.info[split] = {
                'batch_size': loader.batch_size,
                'samples': len(loader.sampler),
                'strategy': loader.sampler.__class__.__name__,
                'drop_last': loader.drop_last,
                'max_epochs': self._lightning_module.trainer.max_epochs,
            }

    @override
    def stop(self, action_name: str) -> None:
        """End recording.

        Args:
            action_name: name of the action that is being profiled
        """
        self.end_time[action_name] = time.perf_counter()
        self.action_count[action_name] += 1
        self.action_total_time[action_name] += (
            self.end_time[action_name] - self.start_time[action_name]
        )

    @override
    def summary(self) -> str:
        """Print summary.

        Returns:
             summary table containing split, strategy, drop last, number of samples, time (s), and sampling rate (samples/s)
        """
        res = '\nProfile Summary \n'
        res += f'\n| {"Split":<10} | {"Strategy":<20} | {"Drop last":<10} | {"Samples":<10} | {"Time (s)":<10} | {"Rate (samples/s)":<16} |'
        res += f'\n| {":":-<10} | {":":-<20} | {":":-<10} | {":":->10} | {":":->10} | {":":->16} |'
        for action_name in self.action_count:
            split = (
                'train'
                if 'train_dataloader_next' in action_name
                else 'val'
                if 'val_next' in action_name
                else None
            )
            if split:
                total_time = self.action_total_time[action_name]
                drop_last = 'True' if self.info[split]['drop_last'] == 1 else 'False'
                if drop_last == 'True':
                    num_batches = (
                        self.action_count[action_name] // self.info[split]['max_epochs']
                        if self.info[split]['max_epochs'] > 0
                        else 0
                    )
                    samples = num_batches * self.info[split]['batch_size']
                else:
                    samples = self.info[split]['samples']
                total_samples = samples * self.info[split]['max_epochs']
                rate = 0.0 if total_time == 0 else total_samples / total_time
                split_name = 'Train' if split == 'train' else 'Validation'
                res += f'\n| {split_name:<10} | {self.info[split]["strategy"]:<20} | {drop_last:<10} | {total_samples:>10} | {total_time:>10.3f} | {rate:>16.3f} |'

        return res
