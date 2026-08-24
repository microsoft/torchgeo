# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
"""A custom I/O Profiler."""

import time
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
        batch_size: int = 32,
    ) -> None:
        """Initialise profiler.

        Args:
            dirpath: root directory to save profiler's results
            filename: name of the file where the profiler's results will be saved
            batch_size: batch size used by data loader
        """
        super().__init__(dirpath=dirpath, filename=filename)
        self.start_time = {}
        self.action_count = defaultdict(int)
        self.batch_size = batch_size
        self.end_time = {}
        self.action_total_time = defaultdict(float)

    @override
    def start(self, action_name: str) -> None:
        """Start recording.

        Args:
            action_name: name of the action that should be profiled
        """
        self.start_time[action_name] = time.perf_counter()

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
             summary table containing split, strategy, number of samples, time (s), and sampling rate (samples/s)
        """
        res = '\nProfile Summary \n'
        res += f'\n| {"Split":<10} | {"Strategy":<10} | {"Samples":<10} | {"Time (s)":<10} | {"Rate (samples/s)":<16} |'
        res += f'\n| {":":-<10} | {":":-<10} | {":":->10} | {":":->10} | {":":->16} |'
        for action_name in self.action_count:
            train = 'train_dataloader_next' in action_name
            val = 'val_next' in action_name
            if train or val:
                total_time = self.action_total_time[action_name]
                action_count = self.action_count[action_name]
                samples = action_count * self.batch_size
                rate = 0.0 if total_time == 0 else samples / total_time
                split = 'Train' if train else 'Validation'
                strategy = 'Random' if train else 'Grid'
                res += f'\n| {split:<10} | {strategy:<10} | {samples:>10} | {total_time:>10.3f} | {rate:>16.3f} |'

        return res
