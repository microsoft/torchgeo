# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LSTM Sequence to Sequence (Seq2Seq) Model."""

import torch
import torch.nn as nn
from torch import Tensor


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x: Tensor):
        # Only keep the last hidden and cell states
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        target_indices: list[int] | None,
        num_layers: int = 1,
        output_sequence_len: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        self.target_indices = target_indices
        self.output_sequence_len = output_sequence_len

    def forward(self, inputs: Tensor, hidden: Tensor, cell: Tensor) -> Tensor:
        # shouldn't this be shape[0] since batch_first = True?
        batch_size = inputs.shape[0]
        outputs = torch.zeros(batch_size, self.output_sequence_len, self.output_size)

        curr_input = inputs[:, 0:1, :]

        for t in range(self.output_sequence_len):
            print(f'input_t: {curr_input.shape}')
            _, (hidden, cell) = self.lstm(curr_input, (hidden, cell))
            output = self.fc(hidden)  # Predict next step
            outputs[:, t, :] = output
            curr_input = output

        return outputs


class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        input_size_encoder: int,
        input_size_decoder: int,
        hidden_size: int = 1,
        output_size: int = 1,
        output_seq_length: int = 1,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(input_size_encoder, hidden_size, num_layers)
        self.decoder = LSTMDecoder(
            input_size_decoder,
            hidden_size,
            output_size,
            target_indices=None,
            num_layers=num_layers,
            output_sequence_len=output_seq_length,
        )

    def forward(self, inputs_encoder: Tensor, inputs_decoder: Tensor) -> Tensor:
        hidden, cell = self.encoder(inputs_encoder)
        outputs = self.decoder(inputs_decoder, hidden, cell)
        return outputs
