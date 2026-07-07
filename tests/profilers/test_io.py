# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import time

from torchgeo.profilers.io import IOBenchProfiler


class TestIOProfiler:
    def test_profiler(self) -> None:
        profiler = IOBenchProfiler(batch_size=32)
        profiler.start('train_dataloader_next')
        time.sleep(0.1)
        profiler.stop('train_dataloader_next')
        profiler.start('val_next')
        time.sleep(0.1)
        profiler.stop('val_next')
        summary = profiler.summary()
        assert 'Train (random)' in summary
        assert 'Validation (grid)' in summary

    def test_zero_time(self) -> None:
        profiler = IOBenchProfiler(batch_size=32)
        profiler.start('train_dataloader_next')
        profiler.stop('train_dataloader_next')
        profiler.start('val_next')
        profiler.stop('val_next')
        profiler._action_total_time['train_dataloader_next'] = 0
        profiler._action_total_time['val_next'] = 0
        summary = profiler.summary()
        assert '0.00000' in summary

    def test_teardown(self) -> None:
        profiler = IOBenchProfiler(batch_size=32)
        profiler.start('train_dataloader_next')
        profiler.stop('train_dataloader_next')
        profiler.teardown(stage='fit')
        assert profiler._action_count == {}
        assert profiler._start == {}
        assert profiler._batch_size == 32
        assert profiler._end == {}
        assert profiler._action_total_time == {}
