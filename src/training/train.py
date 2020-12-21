import torch
from torch import optim
import torch.nn.functional as F
import wandb
from model_zoo.models import get_model
from utils import telegram_bot
import pytorch_lightning as pl


class LitModelClass(pl.LightningModule):
    def __init__(self, name: str, num_classes, config={}):
        super().__init__()
        self.config = config
        self.name = name
        self.num_classes = num_classes
        self.accuracy = pl.metrics.Accuracy()
        self.model = get_model(self.name, self.num_classes)

    # useful only when doing inference
    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        # send a notification
        message = 'Your training has started 🔥 ✨'
        telegram_bot(message)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        acc = self.accuracy(torch.argmax(y_hat, dim=1), y)
        pbar = {'train_acc': acc}

        # log the results to wandb
        wandb.log({'train_loss': loss, 'train_acc': acc})

        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss']
                                     for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['train_acc']
                                    for x in val_step_outputs]).mean()

        pbar = {'val_acc': avg_val_acc}

        # send a notification
        message = f'Validation epoch is over, val_acc: ✨ {avg_val_acc} ✨'
        telegram_bot(message)

        # log the results to wandb
        wandb.log({'val_loss': avg_val_loss, 'val_acc': avg_val_acc})

        return {'val_loss': avg_val_loss, 'progress_bar': pbar}

    def configure_optimizers(self):
        # optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optim.Adam(self.model.parameters(),
                          lr=self.config['lr'])
