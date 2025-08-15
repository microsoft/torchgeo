# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CNN-LSTM Model Architecture."""

from typing import Any

import timm
import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN backbone using a TIMM model.

    Args:
        model_name (str): Name of the TIMM model architecture to use.
        pretrained (bool): Whether to load pretrained weights.
        output_dim (int, optional): Desired output feature dimension. If None, uses
            the TIMM model's native feature dimension.
        freeze_backbone (bool): If True, freezes the backbone parameters.
        in_chans (int): Number of input channels (e.g., 3 for RGB).

    Raises:
        ImportError: If TIMM library is not available.

    Forward:
        x (torch.Tensor): Input tensor of shape either
            - (B, T, C, H, W) for sequences of images, or
            - (B, C, H, W) for single images.

    Returns:
        torch.Tensor: Extracted features of shape (B, T, output_dim) or (B, output_dim).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        output_dim: int | None = None,
        freeze_backbone: bool = False,
        in_chans: int = 3,
    ) -> None:
        """Initialize CNN backbone with TIMM model.

        Args:
            model_name: Name of the TIMM model architecture to use.
            pretrained: Whether to load pretrained weights.
            output_dim: Desired output feature dimension. If None, uses
                the TIMM model's native feature dimension.
            freeze_backbone: If True, freezes the backbone parameters.
            in_chans: Number of input channels (e.g., 3 for RGB).
        """
        super().__init__()

        # Create TIMM model without classification head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )

        # Determine feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, in_chans, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Set output dimension
        if output_dim is None:
            output_dim = feature_dim
        self.output_dim = output_dim

        # Projection layer if needed
        self.projection: nn.Linear | nn.Identity
        if feature_dim != output_dim:
            self.projection = nn.Linear(feature_dim, output_dim)
        else:
            self.projection = nn.Identity()

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN backbone.

        Args:
            x: Input tensor of shape (B, T, C, H, W) for sequences or (B, C, H, W) for single images.

        Returns:
            Extracted features of shape (B, T, output_dim) or (B, output_dim).
        """
        if x.dim() == 5:  # (B, T, C, H, W)
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            features = self.backbone(x)
            features = self.projection(features)
            features = features.view(b, t, self.output_dim)
        else:  # (B, C, H, W)
            features = self.backbone(x)
            features = self.projection(features)
        return features  # type: ignore[no-any-return]


class LSTM(nn.Module):
    """LSTM encoder for sequential data.

    Args:
        input_dim (int): Number of expected features in the input.
        hidden_dim (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
        dropout (float): Dropout probability for LSTM layers (except last).
        batch_first (bool): If True, input and output tensors are provided
            as (batch, seq, feature).

    Forward:
        x (torch.Tensor): Input tensor of shape (B, T, input_dim).
        lengths (torch.Tensor, optional): Lengths of sequences for packing.

    Returns:
        torch.Tensor: Output features from the LSTM of shape (B, T, hidden_dim * num_directions).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        batch_first: bool = True,
    ) -> None:
        """Initialize LSTM encoder.

        Args:
            input_dim: Number of expected features in the input.
            hidden_dim: Number of features in the hidden state.
            num_layers: Number of recurrent layers.
            bidirectional: If True, becomes a bidirectional LSTM.
            dropout: Dropout probability for LSTM layers (except last).
            batch_first: If True, input and output tensors are provided
                as (batch, seq, feature).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=batch_first,
        )

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through LSTM encoder.

        Args:
            x: Input tensor of shape (B, T, input_dim).
            lengths: Lengths of sequences for packing (optional).

        Returns:
            Output features from the LSTM of shape (B, T, hidden_dim * num_directions).
        """
        if lengths is not None:
            # Pack for variable-length sequences
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_x)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=self.batch_first
            )
        else:
            output, _ = self.lstm(x)
        return output


class Task(nn.Module):
    """Unified task head supporting classification and regression.

    Args:
        input_dim (int): Input feature dimension.
        task_type (str): Task type, either 'classification' or 'regression'.
        num_classes (int, optional): Number of classes for classification.
            Required if task_type='classification'.
        output_dim (int): Output dimension for regression (default 1).
        dropout (float): Dropout rate applied before the output layer.
        pooling (str): Method to pool sequence inputs into a single vector.
            Options: 'last', 'mean', 'max', 'attention'.

    Forward:
        x (torch.Tensor): Input tensor, either:
            - (B, input_dim) for single vector, or
            - (B, T, input_dim) for sequences.

    Returns:
        Dict[str, Optional[torch.Tensor]]:
            For classification:
                - 'outputs': Raw logits tensor (B, num_classes).
                - 'probs': Softmax probabilities tensor (B, num_classes).
            For regression:
                - 'outputs': Regression predictions (B, output_dim).
                - 'probs': None.
    """

    def __init__(
        self,
        input_dim: int,
        task_type: str,
        num_classes: int | None = None,
        output_dim: int = 1,
        dropout: float = 0.0,
        pooling: str = 'last',
    ) -> None:
        """Initialize task head.

        Args:
            input_dim: Input feature dimension.
            task_type: Task type, either 'classification' or 'regression'.
            num_classes: Number of classes for classification.
                Required if task_type='classification'.
            output_dim: Output dimension for regression (default 1).
            dropout: Dropout rate applied before the output layer.
            pooling: Method to pool sequence inputs into a single vector.
                Options: 'last', 'mean', 'max', 'attention'.
        """
        super().__init__()
        assert task_type in ('classification', 'regression'), (
            "task_type must be 'classification' or 'regression'"
        )
        self.task_type = task_type
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)

        if self.pooling == 'attention':
            self.attention = nn.Linear(input_dim, 1)

        if task_type == 'classification':
            assert num_classes is not None, (
                'num_classes must be specified for classification'
            )
            self.head = nn.Linear(input_dim, num_classes)
        else:  # regression
            self.head = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Forward pass through task head.

        Args:
            x: Input tensor, either (B, input_dim) for single vector or
               (B, T, input_dim) for sequences.

        Returns:
            Dictionary containing:
                For classification:
                    - 'outputs': Raw logits tensor (B, num_classes).
                    - 'probs': Softmax probabilities tensor (B, num_classes).
                For regression:
                    - 'outputs': Regression predictions (B, output_dim).
                    - 'probs': None.
        """
        # Apply pooling if input is sequence (B, T, input_dim)
        if x.dim() == 3:
            if self.pooling == 'last':
                x = x[:, -1, :]
            elif self.pooling == 'mean':
                x = x.mean(dim=1)
            elif self.pooling == 'max':
                x = x.max(dim=1)[0]
            elif self.pooling == 'attention':
                attn_weights = torch.softmax(self.attention(x), dim=1)
                x = (x * attn_weights).sum(dim=1)

        x = self.dropout(x)
        outputs = self.head(x)

        if self.task_type == 'classification':
            probs = torch.softmax(outputs, dim=-1)
            return {'outputs': outputs, 'probs': probs}
        else:  # regression
            return {'outputs': outputs, 'probs': None}


class CNNLSTM(nn.Module):
    """Combined CNN + LSTM model with a unified task head.

    Args:
        cnn_backbone (CNN): CNN feature extractor module.
        rnn_encoder (LSTM): LSTM sequence encoder module.
        task_head (Task): Task head for classification or regression.

    Raises:
        ValueError: If the output dimension of CNN does not match input dimension of LSTM,
                    or if the output dimension of LSTM does not match input dimension of task head.

    Forward:
        x (torch.Tensor): Input tensor, either
            - (B, T, C, H, W) sequence of images, or
            - (B, C, H, W) single image.
        lengths (torch.Tensor, optional): Lengths of sequences for LSTM packing.

    Returns:
        Dict[str, Optional[torch.Tensor]]: Output dictionary from task head.
    """

    def __init__(self, cnn_backbone: CNN, rnn_encoder: LSTM, task_head: Task) -> None:
        """Initialize CNN-LSTM model.

        Args:
            cnn_backbone: CNN feature extractor module.
            rnn_encoder: LSTM sequence encoder module.
            task_head: Task head for classification or regression.

        Raises:
            ValueError: If the output dimension of CNN does not match input dimension of LSTM,
                        or if the output dimension of LSTM does not match input dimension of task head.
        """
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.rnn_encoder = rnn_encoder
        self.task_head = task_head

        # Validate CNN -> RNN dimensions
        if cnn_backbone.output_dim != rnn_encoder.input_dim:
            raise ValueError(
                f'CNN output dim ({cnn_backbone.output_dim}) must match '
                f'RNN input dim ({rnn_encoder.input_dim})'
            )

        rnn_output_dim = rnn_encoder.output_dim
        task_input_dim = task_head.head.in_features

        if rnn_output_dim != task_input_dim:
            raise ValueError(
                f'RNN output dim ({rnn_output_dim}) must match '
                f'Task input dim ({task_input_dim})'
            )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None, **kwargs: Any
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass through the complete CNN-LSTM model.

        Args:
            x: Input tensor, either (B, T, C, H, W) sequence of images or
               (B, C, H, W) single image.
            lengths: Lengths of sequences for LSTM packing (optional).
            **kwargs: Additional keyword arguments.

        Returns:
            Output dictionary from task head containing predictions and probabilities.
        """
        # Handle single image input by adding time dimension
        if x.dim() == 4:  # (B, C, H, W)
            x = x.unsqueeze(1)  # (B, 1, C, H, W)

        # CNN feature extraction (supports batching all frames)
        cnn_features = self.cnn_backbone(x)  # (B, T, feature_dim)

        # Sequence modeling with LSTM
        rnn_output = self.rnn_encoder(cnn_features, lengths)  # (B, T, hidden_dim)

        # Task-specific output

        output = self.task_head(rnn_output)

        return output  # type: ignore[no-any-return]
