import torch
from torch import optim
import torch.nn.functional as F

from model_zoo.models import get_model
from params import LR
import pytorch_lightning as pl


class LitModelClass(pl.LightningModule):
    def __init__(self, name: str, num_classes):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.accuracy = pl.metrics.Accuracy()
        self.model = get_model(self.name, self.num_classes)

    # useful only when doing inference
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        # acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        # pbar = {"train_acc": acc}
        return {"loss": loss}  # , "progress_bar": pbar}

    # def validation_step(self, batch, batch_idx):
    #     results = self.training_step(batch, batch_idx)
    #     results['progress_bar']['val_acc'] = results['progress_bar']
    #     ['train_acc']

    #     del results['progress_bar']['train_acc']

    #     return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss']
                                     for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc']
                                    for x in val_step_outputs]).mean()

        pbar = {"val_acc": avg_val_acc}
        return {'val_loss': avg_val_loss, "progress_bar": pbar}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=LR)
