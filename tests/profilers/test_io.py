# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import time

import pytest

from torchgeo.profilers.io import IOBenchProfiler


class TestProfiler:
    @pytest.mark.parametrize('name', ['io'])
    def test_profiler(self, name: str) -> None:
        profiler = IOBenchProfiler()
        profiler.start('train_dataloader_next')
        time.sleep(0.1)
        profiler.stop('train_dataloader_next')
        profiler.start('val_next')
        time.sleep(0.1)
        profiler.stop('val_next')
        summary = profiler.summary()
        assert 'Training (random)' in summary
        assert 'Validation (grid)' in summary

    def test_zero_time(self) -> None:
        profiler = IOBenchProfiler()
        profiler.start('train_dataloader_next')
        profiler.stop('train_dataloader_next')
        profiler.start('val_next')
        profiler.stop('val_next')
        profiler._action_total['train_dataloader_next'] = 0.00000
        profiler._action_total['val_next'] = 0.00000
        summary = profiler.summary()
        assert '0.00000' in summary

    def test_teardown(self) -> None:
        profiler = IOBenchProfiler()
        profiler.start('train_dataloader_next')
        profiler.stop('train_dataloader_next')
        profiler.teardown(stage='fit')
        assert len(profiler._action_count) == 0
