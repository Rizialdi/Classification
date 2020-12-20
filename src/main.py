from data.dataModule import LitDataClass
from params import NUM_CLASSES, BATCH_SIZE, EPOCHS, SEED, DEV_MODE
from training.train import LitModelClass
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import datetime
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == "__main__":
    # pretrained model's name
    selected_model = "efficientnet-b1"

    # seed everything for reproducibility
    pl.seed_everything(seed=SEED)

    # create checkpoint folder
    TODAY = str(datetime.date.today())
    DIR = f'../output/checkpoints/{TODAY}'

    if not os.path.exists(DIR):
        os.mkdir(DIR)

    # checkpoint
    checkpoint = ModelCheckpoint(
        dirpath=DIR,
        filename=f'{selected_model}-' +
        '{val_acc:.5f}-' + '{epoch: 02d}',
        monitor='val_acc',
        mode='max')

    # early stopping
    es = EarlyStopping(monitor='val_acc', patience=3, mode='max')

    # logging using Weight&Biaises
    wandb_logger = WandbLogger(project='cassava_land', tags=[
                               selected_model],
                               offline=DEV_MODE,
                               config={'extra': 1})

    # modelModule
    cassava_model = LitModelClass(selected_model, num_classes=NUM_CLASSES)

    # dataModule
    cassava_data = LitDataClass(
        fold=0,
        subset=0.1,
        batch_size=BATCH_SIZE,
        train_val_split=0.9,
        use_extra_data=True)

    trainer = pl.Trainer(deterministic=True,
                         gpus=-1,
                         max_epochs=EPOCHS,
                         callbacks=[es, checkpoint],
                         logger=wandb_logger,
                         fast_dev_run=DEV_MODE)

    trainer.fit(model=cassava_model,
                datamodule=cassava_data)
