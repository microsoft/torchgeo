# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import LSTMSeq2Seq

BATCH_SIZE = [1, 2]
INPUT_SIZE_ENCODER = [1, 3]
INPUT_SIZE_DECODER = [1]
OUTPUT_SIZE = [1]


class TestLSTMSeq2Seq:
    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('e', INPUT_SIZE_ENCODER)
    @pytest.mark.parametrize('d', INPUT_SIZE_DECODER)
    def test_input_size(self, b: int, e: int, d: int) -> None:
        sequence_length = 3
        output_sequence_length = 1
        output_size = 1
        model = LSTMSeq2Seq(
            input_size_encoder=e,
            input_size_decoder=d,
            output_size=output_size,
            output_seq_length=output_sequence_length,
        )
        inputs_encoder = torch.randn(b, sequence_length, e)
        inputs_decoder = torch.randn(b, output_sequence_length, d)
        y = model(inputs_encoder, inputs_decoder)
        assert y.shape == (b, output_sequence_length, output_size)
