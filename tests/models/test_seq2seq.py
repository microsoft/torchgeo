# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Literal

import pytest
import torch

from torchgeo.models import Seq2Seq

BATCH_SIZE = [1, 2, 7]
INPUT_SIZE_ENCODER = [1, 3]
INPUT_SIZE_DECODER = [2, 3]
OUTPUT_SIZE = [1, 2, 3]
NUM_LAYERS = [1, 2, 3]
HIDDEN_SIZE = [1, 2, 3]
RNN_TYPE = {'rnn': torch.nn.RNN, 'gru': torch.nn.GRU, 'lstm': torch.nn.LSTM}


class TestSeq2Seq:
    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('e', INPUT_SIZE_ENCODER)
    @pytest.mark.parametrize('d', INPUT_SIZE_DECODER)
    def test_input_size(self, b: int, e: int, d: int) -> None:
        sequence_length = 3
        output_sequence_length = 3
        n_features = 5
        output_size = 2
        model = Seq2Seq(
            input_size_encoder=e,
            input_size_decoder=d,
            target_indices=list(range(0, output_size)),
            encoder_indices=list(range(0, e)),
            decoder_indices=list(range(0, d)),
            output_size=output_size,
            output_sequence_len=output_sequence_length,
        )
        past_steps = torch.randn(b, sequence_length, n_features)
        future_steps = torch.randn(b, output_sequence_length, n_features)
        y = model(past_steps, future_steps)
        assert y.shape == (b, output_sequence_length, output_size)

    @torch.no_grad()
    @pytest.mark.parametrize('n', NUM_LAYERS)
    def test_num_layers(self, n: int) -> None:
        batch_size = 5
        input_size_encoder = 3
        input_size_decoder = 2
        sequence_length = 3
        output_sequence_length = 3
        n_features = 5
        output_size = 2
        model = Seq2Seq(
            input_size_encoder=input_size_encoder,
            input_size_decoder=input_size_decoder,
            target_indices=list(range(0, output_size)),
            encoder_indices=list(range(0, input_size_encoder)),
            decoder_indices=list(range(0, input_size_decoder)),
            output_size=output_size,
            output_sequence_len=output_sequence_length,
            num_layers=n,
        )
        past_steps = torch.randn(batch_size, sequence_length, n_features)
        future_steps = torch.randn(batch_size, output_sequence_length, n_features)
        y = model(past_steps, future_steps)
        assert y.shape == (batch_size, output_sequence_length, output_size)

    @torch.no_grad()
    @pytest.mark.parametrize('h', HIDDEN_SIZE)
    def test_hidden_size(self, h: int) -> None:
        batch_size = 5
        input_size_encoder = 3
        input_size_decoder = 2
        sequence_length = 3
        output_sequence_length = 3
        n_features = 5
        output_size = 2
        model = Seq2Seq(
            input_size_encoder=input_size_encoder,
            input_size_decoder=input_size_decoder,
            target_indices=list(range(0, output_size)),
            encoder_indices=list(range(0, input_size_encoder)),
            decoder_indices=list(range(0, input_size_decoder)),
            output_size=output_size,
            output_sequence_len=output_sequence_length,
            hidden_size=h,
        )
        past_steps = torch.randn(batch_size, sequence_length, n_features)
        future_steps = torch.randn(batch_size, output_sequence_length, n_features)
        y = model(past_steps, future_steps)
        assert y.shape == (batch_size, output_sequence_length, output_size)

    @torch.no_grad()
    def test_none_indices(self) -> None:
        batch_size = 5
        sequence_length = 3
        output_sequence_length = 1
        input_size = 5
        output_size = 1
        model = Seq2Seq(input_size_encoder=input_size, input_size_decoder=input_size)
        past_steps = torch.randn(batch_size, sequence_length, input_size)
        future_steps = torch.randn(batch_size, output_sequence_length, input_size)
        y = model(past_steps, future_steps)
        assert y.shape == (batch_size, output_sequence_length, output_size)

    @torch.no_grad()
    @pytest.mark.parametrize('o', OUTPUT_SIZE)
    def test_output_size(self, o: int) -> None:
        batch_size = 5
        sequence_length = 3
        output_sequence_length = 1
        input_size = 5
        model = Seq2Seq(
            input_size_encoder=input_size, input_size_decoder=input_size, output_size=o
        )
        past_steps = torch.randn(batch_size, sequence_length, input_size)
        future_steps = torch.randn(batch_size, output_sequence_length, input_size)
        y = model(past_steps, future_steps)
        assert y.shape == (batch_size, output_sequence_length, o)

    @pytest.mark.parametrize('rnn_type', RNN_TYPE.keys())
    def test_rnn_type(self, rnn_type: Literal['rnn', 'gru', 'lstm']) -> None:
        model = Seq2Seq(input_size_encoder=1, input_size_decoder=1, rnn_type=rnn_type)
        assert isinstance(model.encoder.rnn, RNN_TYPE[rnn_type])
        assert isinstance(model.decoder.rnn, RNN_TYPE[rnn_type])
