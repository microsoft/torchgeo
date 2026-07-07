# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
"""A custom I/O Profiler."""

import time
from collections import defaultdict

from lightning.pytorch.profilers import Profiler


class IOBenchProfiler(Profiler):
    """A custom I/O Benchmarking Profiler.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dirpath: str | None = None,
        filename: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialise profiler."""
        super().__init__(dirpath=dirpath, filename=filename)
        self._start: dict[str, float] = {}
        self._action_count: defaultdict[str, int] = defaultdict(int)
        self._batch_size = batch_size
        self._end: dict[str, float] = {}
        self._action_total_time: defaultdict[str, float] = defaultdict(float)

    def start(self, action_name: str) -> None:
        """Start recording."""
        self._start[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        """End recording."""
        self._end[action_name] = time.monotonic()
        self._action_count[action_name] += 1
        self._action_total_time[action_name] += (
            self._end[action_name] - self._start[action_name]
        )

    def summary(self) -> str:
        """Print summary."""
        res = '\nProfile Summary: \n'
        res += '\n| Action:| Time(sec):| Rate (patches/sec): |'
        res += '\n| ------------- | ----------- | ----------- |'
        for action_name in self._action_count:
            train = 'train_dataloader_next' in action_name
            val = 'val_next' in action_name
            if train or val:
                total_time = self._action_total_time[action_name]
                action_count = self._action_count[action_name]
                patches_count = action_count * self._batch_size
                if total_time == 0:
                    rate = 0.0
                else:
                    rate = patches_count / total_time
                label = 'Train (random)' if train else 'Validation (grid)'
                res += f'\n| {label:} | {total_time:.5f} | {rate:.5f} |'

        return res

    def teardown(self, stage: str | None) -> None:
        """Post-profiling tear-down."""
        self._start = {}
        self._action_count.clear()
        self._end = {}
        self._action_total_time.clear()
        super().teardown(stage=stage)
