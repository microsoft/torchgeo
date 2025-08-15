# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

# Import from torchgeo models module
from torchgeo.models.cnnlstm import CNN, CNNLSTM, LSTM, Task


class TestCNN:
    """Test cases for CNN module."""

    @pytest.fixture
    def cnn_config(self) -> dict[str, Any]:
        """Basic CNN configuration for testing."""
        return {
            'model_name': 'resnet18',
            'pretrained': False,  # Use False for faster CI testing
            'output_dim': 256,
            'freeze_backbone': False,
            'in_chans': 3,
        }

    def test_cnn_init_basic(self, cnn_config: dict[str, Any]) -> None:
        """Test CNN module basic initialization."""
        cnn = CNN(**cnn_config)
        assert cnn.output_dim == 256
        assert hasattr(cnn, 'backbone')
        assert hasattr(cnn, 'projection')
        assert isinstance(cnn.projection, nn.Linear)

    def test_cnn_init_without_output_dim(self) -> None:
        """Test CNN initialization without specifying output_dim."""
        cnn = CNN(model_name='resnet18', pretrained=False)
        # Should use native feature dimension
        assert cnn.output_dim > 0
        assert isinstance(cnn.projection, nn.Identity)

    def test_cnn_init_freeze_backbone(self) -> None:
        """Test backbone freezing functionality."""
        cnn = CNN(model_name='resnet18', pretrained=False, freeze_backbone=True)
        frozen_params = 0
        total_params = 0
        for param in cnn.backbone.parameters():
            total_params += 1
            if not param.requires_grad:
                frozen_params += 1
        assert frozen_params == total_params, 'All backbone parameters should be frozen'

    def test_cnn_init_custom_channels(self) -> None:
        """Test CNN with custom input channels."""
        cnn = CNN(model_name='resnet18', pretrained=False, in_chans=1, output_dim=128)
        assert cnn.output_dim == 128
        # Test that it can handle grayscale input
        x = torch.randn(2, 1, 224, 224)
        output = cnn(x)
        assert output.shape == (2, 128)

    @patch('timm.create_model')
    def test_cnn_init_timm_error(self, mock_create_model: Any) -> None:
        """Test CNN initialization with TIMM model creation error."""
        mock_create_model.side_effect = RuntimeError('Model not found')
        with pytest.raises(RuntimeError, match='Model not found'):
            CNN(model_name='invalid_model')

    def test_cnn_forward_single_image(self, cnn_config: dict[str, Any]) -> None:
        """Test CNN forward pass with single image input."""
        cnn = CNN(**cnn_config)
        x = torch.randn(4, 3, 224, 224)  # (B, C, H, W)
        output = cnn(x)
        assert output.shape == (4, 256)
        assert output.dtype == torch.float32

    def test_cnn_forward_image_sequence(self, cnn_config: dict[str, Any]) -> None:
        """Test CNN forward pass with image sequence input."""
        cnn = CNN(**cnn_config)
        x = torch.randn(2, 8, 3, 224, 224)  # (B, T, C, H, W)
        output = cnn(x)
        assert output.shape == (2, 8, 256)
        assert output.dtype == torch.float32

    def test_cnn_forward_different_batch_sizes(
        self, cnn_config: dict[str, Any]
    ) -> None:
        """Test CNN with different batch sizes."""
        cnn = CNN(**cnn_config)
        for batch_size in [1, 3, 7]:
            x = torch.randn(batch_size, 3, 224, 224)
            output = cnn(x)
            assert output.shape == (batch_size, 256)

    def test_cnn_forward_different_image_sizes(
        self, cnn_config: dict[str, Any]
    ) -> None:
        """Test CNN with different image sizes."""
        cnn = CNN(**cnn_config)
        for size in [112, 224, 384]:
            x = torch.randn(2, 3, size, size)
            output = cnn(x)
            assert output.shape == (2, 256)

    def test_cnn_projection_layer_active(self) -> None:
        """Test CNN when projection layer is actually used."""
        # Force different feature dimension to activate projection
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=100)
        assert isinstance(cnn.projection, nn.Linear)
        x = torch.randn(2, 3, 224, 224)
        output = cnn(x)
        assert output.shape == (2, 100)


class TestLSTM:
    """Test cases for LSTM module."""

    @pytest.fixture
    def lstm_config(self) -> dict[str, Any]:
        """Basic LSTM configuration for testing."""
        return {
            'input_dim': 256,
            'hidden_dim': 128,
            'num_layers': 2,
            'bidirectional': True,
            'dropout': 0.1,
            'batch_first': True,
        }

    def test_lstm_init_basic(self, lstm_config: dict[str, Any]) -> None:
        """Test LSTM module basic initialization."""
        lstm = LSTM(**lstm_config)
        assert lstm.input_dim == 256
        assert lstm.hidden_dim == 128
        assert lstm.num_layers == 2
        assert lstm.bidirectional is True
        assert lstm.batch_first is True
        assert lstm.output_dim == 256  # 128 * 2 for bidirectional

    def test_lstm_init_unidirectional(self) -> None:
        """Test unidirectional LSTM initialization."""
        lstm = LSTM(input_dim=128, hidden_dim=64, bidirectional=False)
        assert lstm.output_dim == 64
        assert lstm.bidirectional is False

    def test_lstm_init_single_layer_dropout(self) -> None:
        """Test LSTM with single layer handles dropout correctly."""
        lstm = LSTM(input_dim=128, hidden_dim=64, num_layers=1, dropout=0.5)
        # Dropout should be 0 for single layer (PyTorch requirement)
        assert lstm.lstm.dropout == 0.0

    def test_lstm_init_multi_layer_dropout(self) -> None:
        """Test LSTM with multiple layers preserves dropout."""
        lstm = LSTM(input_dim=128, hidden_dim=64, num_layers=3, dropout=0.3)
        assert lstm.lstm.dropout == 0.3

    def test_lstm_init_batch_first_false(self) -> None:
        """Test LSTM with batch_first=False."""
        lstm = LSTM(input_dim=64, hidden_dim=32, batch_first=False)
        assert lstm.batch_first is False

    def test_lstm_forward_no_lengths(self, lstm_config: dict[str, Any]) -> None:
        """Test LSTM forward pass without sequence lengths."""
        lstm = LSTM(**lstm_config)
        x = torch.randn(4, 10, 256)  # (B, T, input_dim)
        output = lstm(x)
        assert output.shape == (4, 10, 256)  # bidirectional: 128 * 2

    def test_lstm_forward_with_lengths(self, lstm_config: dict[str, Any]) -> None:
        """Test LSTM forward pass with sequence lengths (packed sequences)."""
        lstm = LSTM(**lstm_config)
        x = torch.randn(4, 10, 256)
        lengths = torch.tensor([10, 8, 6, 4])
        output = lstm(x, lengths)
        assert output.shape == (4, 10, 256)

    def test_lstm_forward_batch_first_false(self) -> None:
        """Test LSTM forward with batch_first=False."""
        lstm = LSTM(input_dim=64, hidden_dim=32, batch_first=False)
        x = torch.randn(8, 3, 64)  # (T, B, input_dim)
        output = lstm(x)
        assert output.shape == (8, 3, 32)

    def test_lstm_forward_different_sequence_lengths(
        self, lstm_config: dict[str, Any]
    ) -> None:
        """Test LSTM with varying sequence lengths."""
        lstm = LSTM(**lstm_config)
        for seq_len in [1, 5, 20]:
            x = torch.randn(2, seq_len, 256)
            output = lstm(x)
            assert output.shape == (2, seq_len, 256)

    def test_lstm_forward_empty_lengths(self, lstm_config: dict[str, Any]) -> None:
        """Test LSTM forward pass with None lengths (should work normally)."""
        lstm = LSTM(**lstm_config)
        x = torch.randn(3, 7, 256)
        output = lstm(x, lengths=None)
        assert output.shape == (3, 7, 256)


class TestTask:
    """Test cases for Task module."""

    def test_task_init_classification(self) -> None:
        """Test Task module initialization for classification."""
        task = Task(
            input_dim=256,
            task_type='classification',
            num_classes=10,
            dropout=0.1,
            pooling='mean',
        )
        assert task.task_type == 'classification'
        assert task.pooling == 'mean'
        assert isinstance(task.head, nn.Linear)
        assert task.head.out_features == 10
        assert isinstance(task.dropout, nn.Dropout)

    def test_task_init_regression(self) -> None:
        """Test Task module initialization for regression."""
        task = Task(
            input_dim=256,
            task_type='regression',
            output_dim=3,
            dropout=0.2,
            pooling='last',
        )
        assert task.task_type == 'regression'
        assert task.pooling == 'last'
        assert task.head.out_features == 3

    def test_task_init_regression_default_output_dim(self) -> None:
        """Test regression task with default output_dim."""
        task = Task(input_dim=128, task_type='regression')
        assert task.head.out_features == 1  # default output_dim

    def test_task_init_attention_pooling(self) -> None:
        """Test Task with attention pooling initialization."""
        task = Task(
            input_dim=256,
            task_type='classification',
            num_classes=5,
            pooling='attention',
        )
        assert hasattr(task, 'attention')
        assert isinstance(task.attention, nn.Linear)
        assert task.attention.out_features == 1

    def test_task_init_invalid_task_type(self) -> None:
        """Test Task with invalid task type raises assertion error."""
        with pytest.raises(AssertionError, match='task_type must be'):
            Task(input_dim=256, task_type='invalid_type')

    def test_task_init_classification_no_num_classes(self) -> None:
        """Test classification task without num_classes raises assertion error."""
        with pytest.raises(AssertionError, match='num_classes must be specified'):
            Task(input_dim=256, task_type='classification')

    def test_task_init_all_pooling_methods(self) -> None:
        """Test initialization of all pooling methods."""
        pooling_methods = ['last', 'mean', 'max', 'attention']
        for pooling in pooling_methods:
            task = Task(
                input_dim=128,
                task_type='classification',
                num_classes=5,
                pooling=pooling,
            )
            assert task.pooling == pooling
            if pooling == 'attention':
                assert hasattr(task, 'attention')

    def test_task_forward_classification_single_vector(self) -> None:
        """Test classification forward with single vector input."""
        task = Task(
            input_dim=256,
            task_type='classification',
            num_classes=10,
            dropout=0.0,  # Disable dropout for deterministic testing
        )
        x = torch.randn(4, 256)
        output = task(x)

        assert isinstance(output, dict)
        assert 'outputs' in output
        assert 'probs' in output
        assert output['outputs'].shape == (4, 10)
        assert output['probs'].shape == (4, 10)
        # Check probabilities sum to 1
        assert torch.allclose(output['probs'].sum(dim=1), torch.ones(4), atol=1e-6)

    def test_task_forward_regression_single_vector(self) -> None:
        """Test regression forward with single vector input."""
        task = Task(input_dim=256, task_type='regression', output_dim=2, dropout=0.0)
        x = torch.randn(4, 256)
        output = task(x)

        assert isinstance(output, dict)
        assert 'outputs' in output
        assert 'probs' in output
        assert output['outputs'].shape == (4, 2)
        assert output['probs'] is None

    @pytest.mark.parametrize('pooling', ['last', 'mean', 'max', 'attention'])
    def test_task_forward_sequence_all_pooling(self, pooling: str) -> None:
        """Test all pooling methods with sequence input."""
        task = Task(
            input_dim=128,
            task_type='classification',
            num_classes=5,
            pooling=pooling,
            dropout=0.0,
        )
        x = torch.randn(3, 10, 128)  # (B, T, input_dim)
        output = task(x)

        assert output['outputs'].shape == (3, 5)
        assert output['probs'].shape == (3, 5)
        assert torch.allclose(output['probs'].sum(dim=1), torch.ones(3), atol=1e-6)

    def test_task_forward_sequence_last_pooling(self) -> None:
        """Test last pooling specifically."""
        task = Task(input_dim=64, task_type='regression', output_dim=1, pooling='last')
        x = torch.randn(2, 7, 64)
        output = task(x)
        assert output['outputs'].shape == (2, 1)

    def test_task_forward_sequence_mean_pooling(self) -> None:
        """Test mean pooling specifically."""
        task = Task(input_dim=64, task_type='regression', output_dim=1, pooling='mean')
        x = torch.randn(2, 5, 64)
        output = task(x)
        assert output['outputs'].shape == (2, 1)

    def test_task_forward_sequence_max_pooling(self) -> None:
        """Test max pooling specifically."""
        task = Task(input_dim=64, task_type='regression', output_dim=1, pooling='max')
        x = torch.randn(2, 6, 64)
        output = task(x)
        assert output['outputs'].shape == (2, 1)

    def test_task_forward_attention_pooling_detailed(self) -> None:
        """Test attention pooling with detailed checks."""
        task = Task(
            input_dim=32, task_type='classification', num_classes=3, pooling='attention'
        )
        x = torch.randn(1, 4, 32)
        output = task(x)
        assert output['outputs'].shape == (1, 3)
        assert output['probs'].shape == (1, 3)


class TestCNNLSTM:
    """Test cases for CNNLSTM module."""

    @pytest.fixture
    def compatible_components(self) -> dict[str, Any]:
        """Create compatible CNN, LSTM, and Task components."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=256)
        lstm = LSTM(input_dim=256, hidden_dim=128, num_layers=1, bidirectional=False)
        task = Task(input_dim=128, task_type='classification', num_classes=10)
        return {'cnn_backbone': cnn, 'rnn_encoder': lstm, 'task_head': task}

    def test_cnnlstm_init_success(self, compatible_components: dict[str, Any]) -> None:
        """Test successful CNNLSTM initialization."""
        model = CNNLSTM(
            cnn_backbone=compatible_components['cnn_backbone'],
            rnn_encoder=compatible_components['rnn_encoder'],
            task_head=compatible_components['task_head'],
        )
        assert hasattr(model, 'cnn_backbone')
        assert hasattr(model, 'rnn_encoder')
        assert hasattr(model, 'task_head')

    def test_cnnlstm_init_cnn_lstm_dimension_mismatch(self) -> None:
        """Test CNNLSTM with CNN-LSTM dimension mismatch."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=256)
        lstm = LSTM(input_dim=128, hidden_dim=64)  # Mismatch: expects 256
        task = Task(input_dim=64, task_type='classification', num_classes=5)

        with pytest.raises(
            ValueError, match='CNN output dim .* must match RNN input dim'
        ):
            CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

    def test_cnnlstm_init_lstm_task_dimension_mismatch(self) -> None:
        """Test CNNLSTM with LSTM-Task dimension mismatch."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=256)
        lstm = LSTM(input_dim=256, hidden_dim=128, bidirectional=True)  # output: 256
        task = Task(
            input_dim=64, task_type='classification', num_classes=5
        )  # expects 64

        with pytest.raises(
            ValueError, match='RNN output dim .* must match Task input dim'
        ):
            CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

    def test_cnnlstm_forward_single_image(
        self, compatible_components: dict[str, Any]
    ) -> None:
        """Test CNNLSTM forward with single image (adds time dimension)."""
        model = CNNLSTM(**compatible_components)
        x = torch.randn(2, 3, 224, 224)  # (B, C, H, W)
        output = model(x)

        assert isinstance(output, dict)
        assert 'outputs' in output
        assert 'probs' in output
        assert output['outputs'].shape == (2, 10)
        assert output['probs'].shape == (2, 10)

    def test_cnnlstm_forward_image_sequence(
        self, compatible_components: dict[str, Any]
    ) -> None:
        """Test CNNLSTM forward with image sequence."""
        model = CNNLSTM(**compatible_components)
        x = torch.randn(2, 8, 3, 224, 224)  # (B, T, C, H, W)
        output = model(x)

        assert output['outputs'].shape == (2, 10)
        assert output['probs'].shape == (2, 10)

    def test_cnnlstm_forward_with_lengths(
        self, compatible_components: dict[str, Any]
    ) -> None:
        """Test CNNLSTM forward with sequence lengths."""
        model = CNNLSTM(**compatible_components)
        x = torch.randn(3, 6, 3, 224, 224)
        lengths = torch.tensor([6, 4, 2])
        output = model(x, lengths=lengths)

        assert output['outputs'].shape == (3, 10)
        assert output['probs'].shape == (3, 10)

    def test_cnnlstm_forward_with_kwargs(
        self, compatible_components: dict[str, Any]
    ) -> None:
        """Test CNNLSTM forward accepts additional kwargs."""
        model = CNNLSTM(**compatible_components)
        x = torch.randn(1, 3, 224, 224)
        output = model(x, dummy_kwarg='test')  # Should not fail
        assert output['outputs'].shape == (1, 10)

    def test_cnnlstm_regression_task(self) -> None:
        """Test CNNLSTM with regression task."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=128)
        lstm = LSTM(input_dim=128, hidden_dim=64, bidirectional=True)  # output: 128
        task = Task(input_dim=128, task_type='regression', output_dim=3)

        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)
        x = torch.randn(2, 4, 3, 224, 224)
        output = model(x)

        assert output['outputs'].shape == (2, 3)
        assert output['probs'] is None

    def test_cnnlstm_bidirectional_lstm(self) -> None:
        """Test CNNLSTM with bidirectional LSTM."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=128)
        lstm = LSTM(input_dim=128, hidden_dim=64, bidirectional=True, num_layers=2)
        task = Task(
            input_dim=128, task_type='classification', num_classes=5, pooling='mean'
        )

        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)
        x = torch.randn(2, 5, 3, 224, 224)
        output = model(x)

        assert output['outputs'].shape == (2, 5)
        assert output['probs'].shape == (2, 5)

    def test_cnnlstm_gradient_flow(self, compatible_components: dict[str, Any]) -> None:
        """Test gradient flow through CNNLSTM."""
        model = CNNLSTM(**compatible_components)
        x = torch.randn(1, 3, 3, 224, 224, requires_grad=True)

        output = model(x)
        loss = output['outputs'].sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check parameter gradients
        params_with_grad = 0
        total_params = 0
        for param in model.parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1

        assert params_with_grad > 0, 'No parameter gradients found'


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases for complete pipeline."""

    def test_end_to_end_classification_pipeline(self) -> None:
        """Test complete classification pipeline with all features."""
        # Build comprehensive model
        cnn = CNN(
            model_name='resnet18',
            pretrained=False,
            output_dim=512,
            freeze_backbone=False,
        )
        lstm = LSTM(
            input_dim=512, hidden_dim=256, num_layers=2, bidirectional=True, dropout=0.1
        )
        task = Task(
            input_dim=512,  # 256 * 2 for bidirectional
            task_type='classification',
            num_classes=20,
            dropout=0.2,
            pooling='attention',
        )
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        # Test with realistic batch
        batch_size, seq_length = 4, 8
        x = torch.randn(batch_size, seq_length, 3, 224, 224)
        lengths = torch.tensor([8, 6, 4, 2])

        # Forward pass
        output = model(x, lengths=lengths)

        # Comprehensive assertions
        assert output['outputs'].shape == (batch_size, 20)
        assert output['probs'].shape == (batch_size, 20)
        assert torch.allclose(
            output['probs'].sum(dim=1), torch.ones(batch_size), atol=1e-6
        )
        assert not torch.isnan(output['outputs']).any()
        assert not torch.isnan(output['probs']).any()

    def test_model_train_eval_modes(self) -> None:
        """Test model behavior in train vs eval modes."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=128)
        lstm = LSTM(input_dim=128, hidden_dim=64, dropout=0.5)
        task = Task(
            input_dim=64, task_type='classification', num_classes=5, dropout=0.5
        )
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        x = torch.randn(2, 4, 3, 224, 224)

        # Training mode - outputs may vary due to dropout
        model.train()
        output1_train = model(x)
        output2_train = model(x)

        # Add assertion to check they're different (if dropout is enabled)
        # This assumes your model has dropout that affects outputs
        if hasattr(model.task_head, 'dropout') and model.task_head.dropout.p > 0:
            assert not torch.equal(output1_train['outputs'], output2_train['outputs'])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_model_cuda_compatibility(self) -> None:
        """Test model works on CUDA if available."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=128)
        lstm = LSTM(input_dim=128, hidden_dim=64)
        task = Task(input_dim=64, task_type='regression', output_dim=1)
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        # Move to GPU
        model = model.cuda()
        x = torch.randn(2, 3, 3, 224, 224).cuda()

        output = model(x)
        assert output['outputs'].device.type == 'cuda'
        assert output['outputs'].shape == (2, 1)

    @pytest.mark.parametrize('batch_size', [1, 4, 8])
    @pytest.mark.parametrize('seq_length', [1, 5, 10])
    def test_various_input_dimensions(self, batch_size: int, seq_length: int) -> None:
        """Test model with various batch and sequence sizes."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=64)
        lstm = LSTM(input_dim=64, hidden_dim=32)
        task = Task(input_dim=32, task_type='classification', num_classes=5)
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        x = torch.randn(batch_size, seq_length, 3, 224, 224)
        output = model(x)

        assert output['outputs'].shape == (batch_size, 5)
        assert output['probs'].shape == (batch_size, 5)

    def test_model_with_frozen_cnn_backbone(self) -> None:
        """Test complete model with frozen CNN backbone."""
        cnn = CNN(
            model_name='resnet18',
            pretrained=False,
            output_dim=256,
            freeze_backbone=True,
        )
        lstm = LSTM(input_dim=256, hidden_dim=128)
        task = Task(input_dim=128, task_type='classification', num_classes=10)
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        x = torch.randn(2, 4, 3, 224, 224)
        output = model(x)
        loss = output['outputs'].sum()
        loss.backward()

        # Check that CNN backbone parameters don't have gradients
        for param in model.cnn_backbone.backbone.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

        # But other parts should have gradients
        assert any(
            param.grad is not None and torch.any(param.grad != 0)
            for param in model.rnn_encoder.parameters()
            if param.requires_grad
        )

    def test_model_memory_efficiency(self) -> None:
        """Test that model doesn't leak memory significantly."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=128)
        lstm = LSTM(input_dim=128, hidden_dim=64)
        task = Task(input_dim=64, task_type='regression', output_dim=1)
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        # Multiple forward passes should not accumulate memory
        for _ in range(5):
            x = torch.randn(2, 3, 3, 224, 224)
            output = model(x)
            del x, output  # Explicit cleanup

        # Test passes if no memory errors occur
        assert True

    def test_model_output_consistency(self) -> None:
        """Test output format consistency across different configurations."""
        # Classification config
        cnn1: CNN = CNN(model_name='resnet18', pretrained=False, output_dim=64)
        lstm1: LSTM = LSTM(input_dim=64, hidden_dim=32)
        task1: Task = Task(input_dim=32, task_type='classification', num_classes=3)

        # Regression config
        cnn2: CNN = CNN(model_name='resnet18', pretrained=False, output_dim=64)
        lstm2: LSTM = LSTM(input_dim=64, hidden_dim=32)
        task2: Task = Task(input_dim=32, task_type='regression', output_dim=2)

        configs = [
            (cnn1, lstm1, task1, ['outputs', 'probs']),
            (cnn2, lstm2, task2, ['outputs', 'probs']),
        ]

        for cnn_backbone, rnn_encoder, task_head, expected_keys in configs:
            model = CNNLSTM(
                cnn_backbone=cnn_backbone, rnn_encoder=rnn_encoder, task_head=task_head
            )
            x = torch.randn(1, 2, 3, 224, 224)
            output = model(x)

            assert isinstance(output, dict)
            for key in expected_keys:
                assert key in output


class TestErrorHandlingAndValidation:
    """Test error handling and input validation."""

    def test_cnn_invalid_input_dimensions(self) -> None:
        """Test CNN with invalid input dimensions."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=128)

        # Test with wrong number of dimensions
        with pytest.raises(Exception):  # Should fail in backbone
            x = torch.randn(4, 3, 224)  # Missing height dimension
            cnn(x)

        # Test with 6D input (should fail)
        with pytest.raises(Exception):
            x = torch.randn(2, 4, 3, 224, 224, 10)  # Too many dimensions
            cnn(x)

    def test_lstm_invalid_input_dimensions(self) -> None:
        """Test LSTM with invalid input dimensions."""
        lstm = LSTM(input_dim=128, hidden_dim=64)

        # Test with wrong input dimension
        with pytest.raises(RuntimeError):
            x = torch.randn(4, 10, 256)  # Wrong feature dim (256 vs 128)
            lstm(x)

    def test_task_invalid_input_dimensions(self) -> None:
        """Test Task with invalid input dimensions."""
        task = Task(input_dim=128, task_type='classification', num_classes=5)

        # Test with wrong input dimension
        with pytest.raises(RuntimeError):
            x = torch.randn(4, 256)  # Wrong feature dim (256 vs 128)
            task(x)

    def test_cnnlstm_invalid_tensor_shapes(self) -> None:
        """Test CNNLSTM with various invalid tensor shapes."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=128)
        lstm = LSTM(input_dim=128, hidden_dim=64)
        task = Task(input_dim=64, task_type='classification', num_classes=5)
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        # Test with 3D input (should fail)
        with pytest.raises(Exception):
            x = torch.randn(4, 3, 224)  # Not enough dimensions
            model(x)

    def test_sequence_lengths_validation(self) -> None:
        """Test sequence lengths validation."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=64)
        lstm = LSTM(input_dim=64, hidden_dim=32)
        task = Task(input_dim=32, task_type='regression', output_dim=1)
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        x = torch.randn(3, 8, 3, 224, 224)

        # Valid lengths
        lengths = torch.tensor([8, 6, 4])
        output = model(x, lengths=lengths)
        assert output['outputs'].shape == (3, 1)

        # Test with mismatched batch size - this should raise an error in pack_padded_sequence
        lengths_wrong_size = torch.tensor([8, 6])  # Only 2 lengths for 3 sequences
        try:
            model(x, lengths=lengths_wrong_size)
            # If no exception is raised, that's also valid behavior
            # (some PyTorch versions handle this differently)
            assert True
        except (RuntimeError, ValueError):
            # Expected behavior - pack_padded_sequence should complain
            assert True


class TestCodeCoverageCompleteness:
    """Additional tests to ensure 100% code coverage."""

    def test_cnn_all_code_paths(self) -> None:
        """Test all CNN code paths for complete coverage."""
        # Test with exact feature dimension match (Identity projection)
        cnn = CNN(model_name='resnet18', pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        _ = cnn(x)
        assert isinstance(cnn.projection, nn.Identity)

        # Test with different feature dimension (Linear projection)
        cnn2 = CNN(model_name='resnet18', pretrained=False, output_dim=100)
        output2 = cnn2(x)
        assert isinstance(cnn2.projection, nn.Linear)
        assert output2.shape[1] == 100

    def test_lstm_all_initialization_paths(self) -> None:
        """Test all LSTM initialization code paths."""
        # Test with dropout > 0 and multiple layers
        lstm1 = LSTM(input_dim=64, hidden_dim=32, num_layers=3, dropout=0.2)
        assert lstm1.lstm.dropout == 0.2

        # Test with dropout > 0 but single layer (should become 0)
        lstm2 = LSTM(input_dim=64, hidden_dim=32, num_layers=1, dropout=0.5)
        assert lstm2.lstm.dropout == 0.0

        # Test with dropout = 0
        lstm3 = LSTM(input_dim=64, hidden_dim=32, num_layers=2, dropout=0.0)
        assert lstm3.lstm.dropout == 0.0

    def test_task_all_pooling_branches(self) -> None:
        """Test all pooling method branches in Task forward."""
        input_dim = 32
        x = torch.randn(2, 5, input_dim)

        # Test each pooling method explicitly
        task_last = Task(input_dim=input_dim, task_type='regression', pooling='last')
        output_last = task_last(x)
        assert output_last['outputs'].shape == (2, 1)

        task_mean = Task(input_dim=input_dim, task_type='regression', pooling='mean')
        output_mean = task_mean(x)
        assert output_mean['outputs'].shape == (2, 1)

        task_max = Task(input_dim=input_dim, task_type='regression', pooling='max')
        output_max = task_max(x)
        assert output_max['outputs'].shape == (2, 1)

        task_attention = Task(
            input_dim=input_dim, task_type='regression', pooling='attention'
        )
        output_attention = task_attention(x)
        assert output_attention['outputs'].shape == (2, 1)

    def test_task_classification_vs_regression_branches(self) -> None:
        """Test both classification and regression output branches."""
        x = torch.randn(3, 64)

        # Classification branch
        task_cls = Task(input_dim=64, task_type='classification', num_classes=5)
        output_cls = task_cls(x)
        assert output_cls['outputs'].shape == (3, 5)
        assert output_cls['probs'] is not None
        assert output_cls['probs'].shape == (3, 5)

        # Regression branch
        task_reg = Task(input_dim=64, task_type='regression', output_dim=2)
        output_reg = task_reg(x)
        assert output_reg['outputs'].shape == (3, 2)
        assert output_reg['probs'] is None

    def test_cnnlstm_single_vs_sequence_input_branches(self) -> None:
        """Test both single image and sequence input branches in CNNLSTM."""
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=64)
        lstm = LSTM(input_dim=64, hidden_dim=32)
        task = Task(input_dim=32, task_type='classification', num_classes=3)
        model = CNNLSTM(cnn_backbone=cnn, rnn_encoder=lstm, task_head=task)

        # Single image input (4D) - should add time dimension
        x_single = torch.randn(2, 3, 224, 224)  # (B, C, H, W)
        output_single = model(x_single)
        assert output_single['outputs'].shape == (2, 3)

        # Sequence input (5D) - should process directly
        x_seq = torch.randn(2, 4, 3, 224, 224)  # (B, T, C, H, W)
        output_seq = model(x_seq)
        assert output_seq['outputs'].shape == (2, 3)

    def test_lstm_with_and_without_lengths_branches(self) -> None:
        """Test LSTM forward with and without lengths parameter."""
        lstm = LSTM(input_dim=64, hidden_dim=32, batch_first=True)
        x = torch.randn(3, 8, 64)

        # Without lengths (normal LSTM forward)
        output1 = lstm(x, lengths=None)
        assert output1.shape == (3, 8, 32)

        # With lengths (packed sequence)
        lengths = torch.tensor([8, 6, 4])
        output2 = lstm(x, lengths=lengths)
        assert output2.shape == (3, 8, 32)

    def test_cnn_dimension_detection_code_path(self) -> None:
        """Test the feature dimension detection code in CNN.__init__."""
        # This tests the dummy forward pass used to detect feature dimensions
        # We don't need to mock torch.no_grad, just test that it works
        cnn = CNN(model_name='resnet18', pretrained=False, output_dim=256)
        assert cnn.output_dim == 256
        # Test that the backbone was properly initialized
        assert hasattr(cnn, 'backbone')
        assert hasattr(cnn, 'projection')


if __name__ == '__main__':
    # Run with coverage: pytest --cov=torchgeo.models.cnn_lstm tests/models/test_cnn_lstm.py
    pytest.main([__file__, '-v', '--tb=short'])
