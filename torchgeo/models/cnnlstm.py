# Licensed under the MIT License.

"""CNN-LSTM Model Architecture."""

from typing import Any

import timm
import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN feature extractor using TIMM backbones."""

    def __init__(
        self,
        model_name: str,
        weights: Any = None,
        output_dim: int | None = None,
        in_chans: int = 3,
    ) -> None:
        """Initialize CNN feature extractor.

        Args:
            model_name: Name of the TIMM model architecture to use.
            weights: Weights enum value for loading pretrained weights. 
                If None, no pretrained weights are loaded.
            output_dim: Desired output feature dimension. If None, uses the 
                TIMM model's native feature dimension.
            in_chans: Number of input channels (e.g., 3 for RGB).
        """
        super().__init__()

        # Create TIMM model without classification head
        pretrained = weights is not None
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )

        # Determine feature dimension
        feature_dim = self.backbone.num_features

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN backbone.

        Args:
            x: Input tensor of shape (B, T, C, H, W) for image sequences.

        Returns:
            Extracted features of shape (B, T, output_dim) or (B, output_dim).

        Raises:
            ValueError: If input is not a 5D tensor (sequence of images).
        """
        if x.dim() == 5:  # (B, T, C, H, W)
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            x = self.backbone(x)
            x = self.projection(x)
            x = x.view(b, t, self.output_dim)
        else:  # e.g. (B, C, H, W)
            raise ValueError(
                f"Expected input of shape (B, T, C, H, W), but got shape {tuple(x.shape)} "
                f"with {x.dim()} dimensions. Single images are not supported in this model."
            )
        return x
    
class LSTM(nn.Module):
    """LSTM encoder for sequential data."""

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

class CNNLSTM(nn.Module):
    """Combined CNN + LSTM model for sequence classification/regression."""

    def __init__(
        self,
        model_name: str,
        weights: Any = None,
        cnn_output_dim: int | None = None,
        in_chans: int = 3,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = False,
        lstm_dropout: float = 0.0,
        num_classes: int = 1,
        head_dropout: float = 0.0,
        pooling: str = 'last',
    ) -> None:
        """Initialize CNN-LSTM model.

        Args:
            model_name: Name of the TIMM model architecture to use for CNN backbone.
            weights: Weights enum value for loading pretrained weights.
            cnn_output_dim: CNN output feature dimension. If None, uses native dimension.
            in_chans: Number of input channels (e.g., 3 for RGB).
            lstm_hidden_dim: Number of features in LSTM hidden state.
            lstm_num_layers: Number of recurrent layers in LSTM.
            lstm_bidirectional: If True, use bidirectional LSTM.
            lstm_dropout: Dropout probability for LSTM layers.
            num_classes: Number of output classes/dimensions. Default 1 for
                regression, set to number of classes for classification.
            head_dropout: Dropout rate applied before final layer.
            pooling: Method to pool sequence outputs. Options: 'last', 'mean',
                'max', 'attention'.
        """
        super().__init__()
        self.pooling = pooling

        # Build CNN backbone
        self.cnn_backbone = CNN(
            model_name=model_name,
            weights=weights,
            output_dim=cnn_output_dim,
            in_chans=in_chans,
        )

        # Build LSTM encoder
        self.rnn_encoder = LSTM(
            input_dim=self.cnn_backbone.output_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout,
            batch_first=True,
        )

        # Optional attention for pooling
        if self.pooling == 'attention':
            self.attention = nn.Linear(self.rnn_encoder.output_dim, 1)

        # Head layers
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(self.rnn_encoder.output_dim, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass through the complete CNN-LSTM model.

        Args:
            x: Input tensor of shape (B, T, C, H, W) - sequence of images.
            lengths: Lengths of sequences for LSTM packing (optional).
            **kwargs: Additional keyword arguments.

        Returns:
            Raw logits/predictions of shape (B, num_classes).

        Raises:
            ValueError: If input is not a 5D tensor (sequence of images).
        """
        # Ensure input is a sequence
        if x.dim() != 5:
            raise ValueError(
                f'Expected 5D input tensor (B, T, C, H, W), got {x.dim()}D tensor'
            )

        # CNN feature extraction
        x = self.cnn_backbone(x)  # (B, T, feature_dim)

        # Sequence modeling with LSTM
        x = self.rnn_encoder(x, lengths)  # (B, T, hidden_dim)

        # Pooling
        if self.pooling == 'last':
            x = x[:, -1, :]
        elif self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        elif self.pooling == 'attention':
            attn_weights = torch.softmax(self.attention(x), dim=1)
            x = (x * attn_weights).sum(dim=1)

        # Final prediction
        x = self.dropout(x)
        x = self.head(x)
        return x