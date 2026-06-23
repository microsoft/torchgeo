# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
"""Profiler for I/O benchmark."""

import time
from collections import defaultdict

from lightning.pytorch.profilers import Profiler


class IOBenchProfiler(Profiler):
    """A custom profiler for I/O benchmark.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        dirpath: str | None = None,
        filename: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialise the profiler."""
        super().__init__(dirpath=dirpath, filename=filename)
        self._start = time.monotonic()
        self._action_count: defaultdict[str, int] = defaultdict(int)
        self._action_start: dict[str, float] = {}
        self._action_total: defaultdict[str, float] = defaultdict(float)
        self.batch_size = batch_size

    def start(self, action_name: str) -> None:
        """Start measuring timing for actions."""
        self._action_start[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        """Stop measuring timing for actions."""
        end_time = time.monotonic()
        start_time = self._action_start.get(action_name, end_time)
        self._action_start.pop(action_name, None)
        self._action_total[action_name] += end_time - start_time
        self._action_count[action_name] += 1

    def summary(self) -> str:
        """Print summary of the measurements as a table."""
        duration = time.monotonic() - self._start
        res = '\nProfile Summary\n\n'
        res += f'| {"Action":<19} | {"Time (sec)":>15} | {"Rate (patches/sec)":>20} |\n'
        res += f'| {"-" * 19} | {"-" * 14}: | {"-" * 19}: |\n'
        for action_name in self._action_count:
            train = 'train_dataloader_next' in action_name
            val = 'val_next' in action_name
            if train or val:
                total_time = self._action_total[action_name]
                count = self._action_count[action_name]
                total_patches = count * self.batch_size
                if total_time == 0:
                    rate = 0
                else:
                    rate = total_patches / total_time
                label = 'Training (random)' if train else 'Validation (grid)'
                res += f'| {label:<19} | {total_time:>15.5f} | {rate:>20.5f} |\n'
        res += f'\nTotal duration: {duration:.5f} seconds\n'
        return res

    def teardown(self, stage: str | None) -> None:
        """Reset the profiler."""
        self._action_count.clear()
        self._action_start.clear()
        self._action_total.clear()
        super().teardown(stage=stage)
