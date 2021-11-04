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

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.check_val_every_n_epoch != 0:
            return

        encoder = pl_module.encoder_q

        self.classifier.train()
        for _ in range(self.max_epochs):
            for inputs, targets in self.datamodule.train_dataloader():
                inputs = inputs.to(pl_module.device)
                targets = targets.to(pl_module.device)

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
            inputs = inputs.to(pl_module.device)

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
    def __init__(self, backbone, ftnet, n_classes, freeze_backbone=True, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.finetuner_network = ftnet
        self.freeze_backbone = freeze_backbone
        self.n_classes = n_classes

    def forward(self, x):
        pass

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def shared_step(self, batch):
        pass

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