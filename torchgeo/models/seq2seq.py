# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LSTM Sequence to Sequence (Seq2Seq) Model."""

import random

import torch
import torch.nn as nn
from torch import Tensor


class LSTMEncoder(nn.Module):
    """Encoder for LSTM Seq2Seq."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        """Initialize a new LSTMEncoder.

        Args:
            input_size: The number of features in the input.
            hidden_size: The number of features in the hidden state.
            num_layers: The number of LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the encoder.

        Args:
            x: Input sequence of shape (b, sequence length, input_size).

        Returns:
            Hidden and cell states.
        """
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    """Decoder for LSTM Seq2Seq."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        target_indices: list[int] | None = None,
        num_layers: int = 1,
        output_sequence_len: int = 1,
        teacher_force_prob: float | None = None,
    ) -> None:
        """Initialize a new LSTMDecoder.

        Args:
            input_size: The number of features in the input.
            hidden_size: The number of features in the hidden state.
            output_size: The number of features output by the decoder.
            target_indices: Indices of the target features in the dataset.
                If None, uses all features passed to the decoder. Defaults to None.
            num_layers: Number of LSTM layers. Defaults to 1.
            output_sequence_len: The number of steps to predict forward. Defaults to 1.
            teacher_force_prob: Probability of using teacher forcing. If None, does not
                use teacher forcing. Defaults to None.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_size = output_size
        self.target_indices = target_indices
        self.output_sequence_len = output_sequence_len
        self.teacher_force_prob = teacher_force_prob

    def forward(self, inputs: Tensor, hidden: Tensor, cell: Tensor) -> Tensor:
        """Forward pass of the decoder.

        Args:
            inputs: Input sequence of shape (b, sequence length, input_size).
            hidden: hidden state from the encoder.
            cell: cell state from the encoder.

        Returns:
            Output sequence of shape (b, output_sequence_len, output_size).
        """
        batch_size = inputs.shape[0]
        outputs = torch.zeros(
            batch_size, self.output_sequence_len, self.output_size, device=inputs.device
        )

        current_input = inputs[:, 0:1, :]

        for t in range(self.output_sequence_len):
            _, (hidden, cell) = self.lstm(current_input, (hidden, cell))
            last_layer_hidden = hidden[-1:]
            output = self.fc(last_layer_hidden)
            output = output.permute(1, 0, 2)  # put batch dimension first
            outputs[:, t : t + 1, :] = output
            current_input = inputs[:, t : t + 1, :].clone()
            teacher_force = (
                random.random() < self.teacher_force_prob
                if self.teacher_force_prob is not None
                else False
            )
            if not teacher_force:
                if self.target_indices:
                    current_input[:, :, self.target_indices] = output
                else:
                    current_input = output

        return outputs


class LSTMSeq2Seq(nn.Module):
    """LSTM Sequence-to-Sequence (Seq2Seq)."""

    def __init__(
        self,
        input_size_encoder: int,
        input_size_decoder: int,
        target_indices: list[int] | None = None,
        encoder_indices: list[int] | None = None,
        decoder_indices: list[int] | None = None,
        hidden_size: int = 1,
        output_size: int = 1,
        output_sequence_len: int = 1,
        num_layers: int = 1,
        teacher_force_prob: float | None = None,
    ) -> None:
        """Initialize a new LSTMSeq2Seq model.

        Args:
            input_size_encoder: The number of features in the encoder input.
            input_size_decoder: The number of features in the decoder input.
            target_indices: The indices of the target(s) in the dataset. If None, uses all features. Defaults to None.
            encoder_indices: The indices of the encoder inputs. If None, uses all features. Defaults to None.
            decoder_indices: The indices of the decoder inputs. If None, uses all features. Defaults to None.
            hidden_size: The number of features in the hidden states of the encoder and decoder. Defaults to 1.
            output_size: The number of features output by the model. Defaults to 1.
            output_sequence_len: The number of steps to predict forward. Defaults to 1.
            num_layers: Number of LSTM layers in the encoder and decoder. Defaults to 1.
            teacher_force_prob: Probability of using teacher forcing. If None, does not
                use teacher forcing. Defaults to None.
        """
        super().__init__()
        for indices, size, name in [
            (encoder_indices, input_size_encoder, 'encoder_indices'),
            (decoder_indices, input_size_decoder, 'decoder_indices'),
            (target_indices, output_size, 'target_indices'),
        ]:
            if indices:
                assert len(indices) == size, f'Length of {name} should match {size}.'
        if decoder_indices and isinstance(target_indices, list):
            assert set(target_indices).issubset(set(decoder_indices)), (
                'target_indices should be in decoder_indices.'
            )
            # Target indices need to be mapped to the subset of inputs for decoder
            target_indices = [
                i for i, val in enumerate(decoder_indices) if val in target_indices
            ]
        self.encoder = LSTMEncoder(input_size_encoder, hidden_size, num_layers)
        self.decoder = LSTMDecoder(
            input_size=input_size_decoder,
            hidden_size=hidden_size,
            output_size=output_size,
            target_indices=target_indices,
            num_layers=num_layers,
            output_sequence_len=output_sequence_len,
            teacher_force_prob=teacher_force_prob,
        )
        self.encoder_indices = encoder_indices
        self.decoder_indices = decoder_indices

    def forward(self, past_steps: Tensor, future_steps: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            past_steps: Past time steps.
            future_steps: Future time steps.

        Returns:
            Output sequence of shape (b, output_sequence_len, output_size).
        """
        if self.encoder_indices:
            inputs_encoder = past_steps[:, :, self.encoder_indices]
        else:
            inputs_encoder = past_steps
        inputs_decoder = torch.cat(
            [past_steps[:, -1, :].unsqueeze(1), future_steps], dim=1
        )
        if self.decoder_indices:
            inputs_decoder = inputs_decoder[:, :, self.decoder_indices]
        hidden, cell = self.encoder(inputs_encoder)
        outputs: Tensor = self.decoder(inputs_decoder, hidden, cell)
        return outputs
