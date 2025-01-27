# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for instance segmentation."""

from typing import Any                                            
import torch.nn as nn                                            
import torch                                                     
from torch import Tensor                                         
from torchmetrics.detection.mean_ap import MeanAveragePrecision  
from torchmetrics import MetricCollection
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .base import BaseTask  

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..datasets import RGBBandsMissingError, unbind_samples

class InstanceSegmentationTask(BaseTask):
    """Instance Segmentation."""

    def __init__(
        self,
        model: str = 'mask_rcnn',           
        backbone: str = 'resnet50',         
        weights: str | bool | None = None, 
        num_classes: int = 2,              
        lr: float = 1e-3,                   
        patience: int = 10,                 
        freeze_backbone: bool = False,      
    ) -> None:
        """Initialize a new SemanticSegmentationTask instance.

        Args:
            model: Name of the model to use.
            backbone: Name of the backbone to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False or
                None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes (including the background).
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.

        .. versionadded:: 0.7
        """
        self.weights = weights         
        super().__init__()              
        self.save_hyperparameters()     
        self.model = None               
        self.validation_outputs = []    
        self.test_outputs = []          
        self.configure_models()         
        self.configure_metrics()        

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model = self.hparams['model'].lower()      
        num_classes = self.hparams['num_classes']  

        if model == 'mask_rcnn':
            # Load the Mask R-CNN model with a ResNet50 backbone
            self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)  

            # Update the classification head to predict `num_classes` 
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # self.model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # Update the mask head for instance segmentation
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            self.model.roi_heads.mask_predictor = nn.ConvTranspose2d(
                in_features_mask, num_classes, kernel_size=2, stride=2
            )

        else:
            raise ValueError(
                f"Invalid model type '{model}'. Supported model: 'mask_rcnn'"
            )

        # Freeze backbone 
        if self.hparams['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False  


    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        - Uses Mean Average Precision (mAP) for masks (IOU-based metric).
        """
        self.metrics = MetricCollection([MeanAveragePrecision(iou_type="segm")])
        self.train_metrics = self.metrics.clone(prefix='train_')
        self.val_metrics = self.metrics.clone(prefix='val_')
        self.test_metrics = self.metrics.clone(prefix='test_')

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Compute the training loss.

        Args:
            batch: A batch of data from the DataLoader. Includes images and ground truth targets.
            batch_idx: Index of the current batch.

        Returns:
            The total loss for the batch.
        """
        images, targets = batch['image'], batch['target']     
        loss_dict = self.model(images, targets)               
        loss = sum(loss for loss in loss_dict.values())       
        self.log('train_loss', loss, batch_size=len(images))  
        return loss  

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Compute the validation loss.

        Args:
            batch: A batch of data from the DataLoader. Includes images and targets.
            batch_idx: Index of the current batch.

        Updates metrics and stores predictions/targets for further analysis.
        """
        images, targets = batch['image'], batch['target']   
        batch_size = images.shape[0]
         
        outputs = self.model(images) 

        for target in targets:
            target["masks"] = (target["masks"] > 0).to(torch.uint8)
            target["boxes"] = target["boxes"].to(torch.float32)
            target["labels"] = target["labels"].to(torch.int64)
        
        # Compute the loss and predictions
        loss_dict = self.model(images, targets)  # list of dictionaries

        print('\nDEBUG TRAINING LOSS\n')
        print(f"Training loss: {loss_dict}")

        # Post-process `loss_dict` to compute total loss
        total_loss = 0.0  
        for loss in loss_dict:
            if isinstance(loss, dict):
                for key, value in loss.items():
                    # Ensure the loss component is a scalar tensor
                    if value.ndim == 0: 
                        total_loss += value
                    else:
                        print(f"Skipping non-scalar loss: {key}, shape: {value.shape}")

        # Post-process the outputs to ensure masks are in the correct format
        for output in outputs:
            if "masks" in output:
                output["masks"] = (output["masks"] > 0.5).squeeze(1).to(torch.uint8)
        
        # Sum the losses
        self.log('val_loss', total_loss, batch_size=batch_size)

        metrics = self.val_metrics(outputs, targets)
        # Log only scalar values from metrics
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                # Cast to float if integer and compute mean
                value = value.to(torch.float32).mean()
            scalar_metrics[key] = value

        self.log_dict(scalar_metrics, batch_size=batch_size)               

        # check
        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule

            batch['prediction_masks'] = [output['masks'].cpu() for output in outputs]  
            batch['image'] = batch['image'].cpu()

            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )
                plt.close()
    
    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Compute the test loss and additional metrics."""

        images, targets = batch['image'], batch['target']
        batch_size = images.shape[0]

        outputs = self.model(images)

        print('\nDEBUG THE PREDICTIONS\n')
        print(f"Predictions for batch {batch_idx}: {outputs}")
        print(f"Ground truth for batch {batch_idx}: {targets}")

        for target in targets:
            target["masks"] = target["masks"].to(torch.uint8)
            target["boxes"] = target["boxes"].to(torch.float32)
            target["labels"] = target["labels"].to(torch.int64)

        for output in outputs:
            output["masks"] = (output["masks"] > 0.5).squeeze(1).to(torch.uint8)

        loss_dict = self.model(images, targets)  # Compute all losses 

        # Post-process `loss_dict` to compute total loss
        total_loss = 0.0  
        for loss in loss_dict:
            if isinstance(loss, dict):
                for key, value in loss.items():
                    # Ensure the loss component is a scalar tensor
                    if value.ndim == 0: 
                        total_loss += value
                    else:
                        print(f"Skipping non-scalar loss: {key}, shape: {value.shape}")


        # Sum the losses
        self.log('test_loss', total_loss, batch_size=batch_size)

        metrics = self.val_metrics(outputs, targets)
        # Log only scalar values from metrics
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                # Cast to float if integer and compute mean
                value = value.to(torch.float32).mean()
            scalar_metrics[key] = value

        self.log_dict(scalar_metrics, batch_size=batch_size)

        print('\nDEBUG CAL METRICS\n')
        print(f"Validation metrics: {metrics}")

        return outputs

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        """Perform inference on a batch of images.

        Args:
            batch: A batch of images.

        Returns:
            Predicted masks and bounding boxes for the batch.
        """
        self.model.eval()
        images = batch['image']           
        outputs = self.model(images) 
        return outputs          








