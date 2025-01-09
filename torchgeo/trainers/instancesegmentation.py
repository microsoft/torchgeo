from typing import Any                                           # Allows us to annotate arguments and return types of functions
import torch.nn as nn                                            # PyTorch module for neural network layers
import torch                                                     # PyTorch for deep learning operations
from torch import Tensor                                         # Type hint for tensors
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # Metric to evaluate instance segmentation models
from torchvision.models.detection import maskrcnn_resnet50_fpn   # Pre-built Mask R-CNN model from PyTorch
from ultralytics import YOLO  
from .base import BaseTask  

class InstanceSegmentationTask(BaseTask):
    """
    Task class for training and evaluating instance segmentation models.

    This class supports Mask R-CNN and YOLO models and handles the following:
    - Model configuration
    - Loss computation
    - Metric computation (e.g., Mean Average Precision)
    - Training, validation, testing, and prediction steps
    """

    def __init__(
        self,
        model: str = 'mask_rcnn',           # Model type, e.g., 'mask_rcnn' or 'yolo'
        backbone: str = 'resnet50',         # Backbone type for Mask R-CNN (ignored for YOLO)
        weights: str | bool | None = None,  # Pretrained weights or custom checkpoint path
        num_classes: int = 2,               # Number of classes, including background
        lr: float = 1e-3,                   # Learning rate for the optimizer
        patience: int = 10,                 # Patience for the learning rate scheduler
        freeze_backbone: bool = False,      # Whether to freeze backbone layers (useful for transfer learning)
    ) -> None:
        """
        Constructor for the InstanceSegmentationTask.

        Initializes the hyperparameters, sets up the model and metrics.
        """
        self.weights = weights          # Save weights for model initialization
        super().__init__()              # Initialize the BaseTask class (inherits common functionality)
        self.save_hyperparameters()     # Save input arguments for later use (e.g., in checkpoints or logs)
        self.model = None               # Placeholder for the model (to be initialized later)
        self.validation_outputs = []    # List to store outputs during validation (used for debugging or analysis)
        self.test_outputs = []          # List to store outputs during testing
        self.configure_models()         # Call method to set up the model
        self.configure_metrics()        # Call method to set up metrics

    def configure_models(self) -> None:
        """
        Set up the instance segmentation model based on the specified type (Mask R-CNN or YOLO).

        Configures:
        - Backbone (for Mask R-CNN)
        - Classifier and mask heads
        - Pretrained weights
        """
        model = self.hparams['model'].lower()      # Read the model type from hyperparameters (convert to lowercase)
        num_classes = self.hparams['num_classes']  # Number of output classes

        if model == 'mask_rcnn':
            # Load the Mask R-CNN model with a ResNet50 backbone
            self.model = maskrcnn_resnet50_fpn(pretrained=self.weights is True)

            # Update the classification head to predict `num_classes` 
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)

            # Update the mask head for instance segmentation
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.model.roi_heads.mask_predictor = nn.ConvTranspose2d(
                in_features_mask, num_classes, kernel_size=2, stride=2
            )

        elif model == 'yolo':
            # Initialize YOLOv8 for instance segmentation
            self.model = YOLO('yolov8n-seg')           # Load a small YOLOv8 segmentation model
            self.model.model.args['nc'] = num_classes  # Set the number of classes in YOLO
            if self.weights:
                # If weights are provided, load the custom checkpoint
                self.model = YOLO(self.weights)

        else:
            raise ValueError(
                f"Invalid model type '{model}'. Supported models: 'mask_rcnn', 'yolo'."
            )

        # Freeze the backbone if specified (useful for transfer learning)
        if self.hparams['freeze_backbone'] and model == 'mask_rcnn':
            for param in self.model.backbone.parameters():
                param.requires_grad = False  # Prevent these layers from being updated during training

    def configure_metrics(self) -> None:
        """
        Set up metrics for evaluating instance segmentation models.

        - Uses Mean Average Precision (mAP) for masks (IOU-based metric).
        """
        self.metrics = MeanAveragePrecision(iou_type="segm")  # Track segmentation-specific mAP

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Args:
            batch: A batch of data from the DataLoader. Includes images and ground truth targets.
            batch_idx: Index of the current batch.

        Returns:
            The total loss for the batch.
        """
        images, targets = batch['image'], batch['target']     # Unpack images and targets
        loss_dict = self.model(images, targets)               # Compute losses (classification, box regression, mask loss, etc.)
        loss = sum(loss for loss in loss_dict.values())       # Combine all losses into a single value
        self.log('train_loss', loss, batch_size=len(images))  # Log the training loss for monitoring
        return loss  # Return the loss for optimization

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        Perform a single validation step.

        Args:
            batch: A batch of data from the DataLoader. Includes images and targets.
            batch_idx: Index of the current batch.

        Updates metrics and stores predictions/targets for further analysis.
        """
        images, targets = batch['image'], batch['target']   # Unpack images and targets
        outputs = self.model(images)                        # Run inference on the model
        self.metrics.update(outputs, targets)               # Update mAP metrics with predictions and ground truths
        self.validation_outputs.append((outputs, targets))  # Store outputs for debugging or visualization

    def on_validation_epoch_end(self) -> None:
        """
        At the end of the validation epoch, compute and log metrics.

        Resets the stored outputs to free memory.
        """
        metrics_dict = self.metrics.compute()   # Calculate final mAP and other metrics
        self.log_dict(metrics_dict)             # Log all computed metrics
        self.metrics.reset()                    # Reset metrics for the next epoch
        self.validation_outputs.clear()         # Clear stored outputs to free memory

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """
        Perform a single test step.

        Similar to validation but used for test data.
        """
        images, targets = batch['image'], batch['target']
        outputs = self.model(images)
        self.metrics.update(outputs, targets)
        self.test_outputs.append((outputs, targets))

    def on_test_epoch_end(self) -> None:
        """
        At the end of the test epoch, compute and log metrics.

        Resets the stored outputs to free memory.
        """
        metrics_dict = self.metrics.compute()
        self.log_dict(metrics_dict)
        self.metrics.reset()
        self.test_outputs.clear()

    def predict_step(self, batch: Any, batch_idx: int) -> Tensor:
        """
        Perform inference on a batch of images.

        Args:
            batch: A batch of images.

        Returns:
            Predicted masks and bounding boxes for the batch.
        """
        images = batch['image']           # Extract images from the batch
        predictions = self.model(images)  # Run inference on the model
        return predictions                # Return the predictions
