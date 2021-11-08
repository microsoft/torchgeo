import torch
from torch import nn
from pytorch_lightning import Callback
import torch
from itertools import chain
from torch import nn, optim
from pytorch_lightning import LightningModule
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, IoU
import segmentation_models_pytorch as smp


class SSLClassificationProbingEvaluator(Callback):

    def __init__(self, z_dim, datamodule, n_classes, max_epochs=10, check_val_every_n_epoch=1, batch_size=1024, num_workers=32):
        self.z_dim = z_dim
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self.datamodule = datamodule
        self.n_classes = n_classes

        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.metric = lambda output, target: Accuracy(target, output) * 100.0

    def on_pretrain_routine_start(self, trainer, pl_module):
        """Use lighting evaluator with n_hidden None --> Same as Linear probing."""
        self.classifier = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.n_classes,
            n_hidden=None
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=1e-3)

    def on_epoch_end(self, trainer, ssl_module):
        if (trainer.current_epoch + 1) % self.check_val_every_n_epoch != 0:
            return

        encoder = ssl_module.encoder

        self.classifier.train()
        for _ in range(self.max_epochs):
            for inputs, targets in self.datamodule.train_dataloader():
                inputs = inputs.to(ssl_module.device)
                targets = targets.to(ssl_module.device)

                with torch.no_grad():
                    representations = encoder(inputs)
                representations = representations.detach()

                logits = self.classifier(representations)
                loss = self.criterion(logits, targets)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.classifier.eval()
        accuracies = []
        for inputs, targets in self.datamodule.val_dataloader():
            inputs = inputs.to(ssl_module.device)

            with torch.no_grad():
                representations = encoder(inputs)
            representations = representations.detach()

            logits = self.classifier(representations)
            preds = torch.sigmoid(logits).detach().cpu()
            acc = self.metric(preds, targets)
            accuracies.append(acc)
        acc = torch.mean(torch.tensor(accuracies))

        metrics = {'online_val_acc': acc}
        trainer.logger_connector.log_metrics(metrics, {})
        trainer.logger_connector.add_progress_bar_metrics(metrics)



class SSLSegmentationFineTunerEvaluator(LightningModule):
    num_filters = 64

    def config_task(self, kwargs: Any) -> None:
        """Configures the task based on kwargs parameters."""
        pretrained = ("imagenet" in self.hparams["weights"]) and not os.path.exists(self.hparams["weights"])
        if kwargs["segmentation_model"] == "unet":
            self.finetuner_model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=None,
                in_channels=self.in_channels,
                classes=self.n_classes,
            )
        elif kwargs["segmentation_model"] == "deeplabv3+":
            self.finetuner_model = smp.DeepLabV3Plus(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=None,
                encoder_output_stride=kwargs["encoder_output_stride"],
                in_channels=self.in_channels,
                classes=self.n_classes,
            )
        elif kwargs["segmentation_model"] == "fcn":
            self.finetuner_model = FCN(self.in_channels, self.n_classes, self.num_filters)
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if os.path.exists(self.hparams["weights"]):
            name, state_dict = utils.extract_encoder(self.hparams["weights"])
            if self.hparams["encoder_name"] != name:
            raise ValueError(
                    f"""Trying to load {name} weights into a"""
                    f"""{self.hparams['encoder_name']}"""
                )
            self.backbone = utils.load_state_dict(self.finetuner_model.encoder, state_dict)

        if kwargs["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()  # type: ignore[attr-defined]
        elif kwargs["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")


    def __init__(self, datamodule, n_classes, in_channels, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.config_task(kwargs)

        self.train_iou = IoU(num_classes=self.n_classes)
        self.val_iou = IoU(num_classes=self.n_classes)
        self.test_iou = IoU(num_classes=self.n_classes)

    def forward(self, x: Tensor) -> Any:
        """Forward pass of the model."""
        return self.finetuner_network(x)

    def on_train_epoch_start(self) -> None:
        #TODO: decide on freezing this or not
        self.backbone.eval()

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        loss = self.shared_step(batch)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_accuracy(y_hat_hard, y)
        self.train_iou(y_hat_hard, y)

        return cast(Tensor, loss)

    def validation_step(# type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        loss = self.shared_step(batch)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss)
        self.val_accuracy(y_hat_hard, y)
        self.val_iou(y_hat_hard, y)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                batch["image"][0].cpu().numpy(), 0, 3
            )
            mask = batch["mask"][0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask, vmin=0, vmax=4)
            axs[1].axis("off")
            axs[2].imshow(pred, vmin=0, vmax=4)
            axs[2].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc", self.val_accuracy.compute())
        self.log("val_iou", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        loss = self.shared_step(batch)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_accuracy(y_hat_hard, y)
        self.test_iou(y_hat_hard, y)
    
    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc", self.test_accuracy.compute())
        self.log("test_iou", self.test_iou.compute())
        self.test_accuracy.reset()
        self.test_iou.reset()

    def shared_step(self, batch):
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss(y_hat, y)

    def configure_optimizers(self):
        params = self.finetuner_network.parameters()
        if not self.freeze_backbone:
            params = chain(self.backbone.parameters(), params)
        optimizer = optim.Adam(params, lr=self.hparams["learning_rate"])
        scheduler = ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}