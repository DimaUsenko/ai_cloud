import pytorch_lightning as pl
import torchmetrics
from F2 import F2
from FRCNNObjectDetector import FRCNNObjectDetector
import torch


class COCO_Module(pl.LightningModule):

    def __init__(self, pretrained_weights_path=None, num_classes=3):
        super().__init__()

        self.model = self._create_model(pretrained_weights_path, num_classes)

        self.val_map = torchmetrics.MAP()
        self.val_f2 = F2()

    def _create_model(self, pretrained_weights_path, num_classes):
        return FRCNNObjectDetector(pretrained_weights_path, num_classes)

    def forward(self, image):
        self.model.eval()
        output = self.model(image)

        return output

    def training_step(self, batch, batch_idx):
        image, target = batch
        loss_dict = self.model(image, target)
        losses = sum(loss for loss in loss_dict.values())

        batch_size = len(batch[0])
        self.log_dict(loss_dict, batch_size=batch_size)
        self.log("train_loss", losses, batch_size=batch_size)

        return losses

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)

        val_map = self.val_map(output, target)
        val_f2 = self.val_f2(output, target)

        self.log("val_map", val_map["map"])
        self.log("val_f2", val_f2)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return [optimizer], [lr_scheduler]
