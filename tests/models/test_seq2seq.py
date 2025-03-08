# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import LSTMSeq2Seq

BATCH_SIZE = [1, 2, 7]
INPUT_SIZE_ENCODER = [1, 3]
INPUT_SIZE_DECODER = [2, 3]
OUTPUT_SIZE = [1]


class TestLSTMSeq2Seq:
    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('e', INPUT_SIZE_ENCODER)
    @pytest.mark.parametrize('d', INPUT_SIZE_DECODER)
    def test_input_size(self, b: int, e: int, d: int) -> None:
        sequence_length = 3
        output_sequence_length = 3
        n_features = 5
        output_size = 2
        model = LSTMSeq2Seq(
            input_size_encoder=e,
            input_size_decoder=d,
            target_indices=list(range(0, output_size)),
            encoder_indices=list(range(0, e)),
            decoder_indices=list(range(0, d)),
            output_size=output_size,
            output_seq_length=output_sequence_length,
        )
        past_steps = torch.randn(b, sequence_length, n_features)
        future_steps = torch.randn(b, output_sequence_length, n_features)
        y = model(past_steps, future_steps)
        assert y.shape == (b, output_sequence_length, output_size)
