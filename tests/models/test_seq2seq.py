# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Literal

import pytest
import torch

from torchgeo.models import Seq2Seq

BATCH_SIZE = [1, 2, 7]
INPUT_SIZE = [1, 3]
NUM_LAYERS = [1, 2, 3]
HIDDEN_SIZE = [1, 2, 3]
RNN_TYPE = {'rnn': torch.nn.RNN, 'gru': torch.nn.GRU, 'lstm': torch.nn.LSTM}


class TestSeq2Seq:
    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('s', INPUT_SIZE)
    def test_input_size(self, b: int, s: int) -> None:
        sequence_length = 3
        output_sequence_length = 3
        model = Seq2Seq(input_size=s, output_sequence_len=output_sequence_length)
        past_targets = torch.randn(b, sequence_length, s)
        future_targets = torch.randn(b, output_sequence_length, s)
        y = model(past_targets, future_targets)
        assert y.shape == (b, output_sequence_length, s)

    @torch.no_grad()
    @pytest.mark.parametrize('n', NUM_LAYERS)
    def test_num_layers(self, n: int) -> None:
        batch_size = 5
        input_size = 2
        sequence_length = 3
        output_sequence_length = 3
        n_features = 2
        output_size = 2
        model = Seq2Seq(
            input_size=input_size,
            output_sequence_len=output_sequence_length,
            num_layers=n,
        )
        past_targets = torch.randn(batch_size, sequence_length, n_features)
        future_targets = torch.randn(batch_size, output_sequence_length, n_features)
        y = model(past_targets, future_targets)
        assert y.shape == (batch_size, output_sequence_length, output_size)

    @torch.no_grad()
    @pytest.mark.parametrize('h', HIDDEN_SIZE)
    def test_hidden_size(self, h: int) -> None:
        batch_size = 5
        input_size = 2
        sequence_length = 3
        output_sequence_length = 3
        n_features = 2
        output_size = 2
        model = Seq2Seq(
            input_size=input_size,
            output_sequence_len=output_sequence_length,
            hidden_size=h,
        )
        past_targets = torch.randn(batch_size, sequence_length, n_features)
        future_targets = torch.randn(batch_size, output_sequence_length, n_features)
        y = model(past_targets, future_targets)
        assert y.shape == (batch_size, output_sequence_length, output_size)

    @pytest.mark.parametrize('rnn_type', RNN_TYPE.keys())
    def test_rnn_type(self, rnn_type: Literal['rnn', 'gru', 'lstm']) -> None:
        model = Seq2Seq(input_size=1, rnn_type=rnn_type)
        assert isinstance(model.encoder.rnn, RNN_TYPE[rnn_type])
        assert isinstance(model.decoder.rnn, RNN_TYPE[rnn_type])

    def test_no_future_targets(self) -> None:
        model = Seq2Seq(input_size=1, output_sequence_len=3)
        past_target = torch.rand(1, 10, 1)
        model(past_target)
