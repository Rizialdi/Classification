import os
import wandb
import datetime
import pytorch_lightning as pl
from training.train import LitModelClass
from data.dataModule import LitDataClass
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from params import BATCH_SIZE, EPOCHS, SEED, DROPOUT, DEV_MODE, SUBSET
from params import IMAGE_SIZE, NUM_CLASSES, LR, FOLD

if __name__ == "__main__":
    # hyperparameters
    hyperparameter_defaults = dict(
        img_size=IMAGE_SIZE,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        lr=LR,
        use_extra_data=True,
        epochs=EPOCHS,
        subset=SUBSET,
        fold=FOLD
    )

    # pretrained model's name
    selected_model = "efficientnet-b1"

    # seed everything for reproducibility
    pl.seed_everything(seed=SEED)

    # create checkpoint folder
    TODAY = str(datetime.date.today())
    DIR = f'../output/checkpoints/{TODAY}'

    if not os.path.exists(DIR):
        os.mkdir(DIR)

    # logging using Weight&Biaises
    experiment = wandb.init(project="cassava_land",
                            tags=[selected_model],
                            config=hyperparameter_defaults)

    # make sure config dict come from
    # the set of values in experiment
    config = experiment.config

    wandb_logger = WandbLogger(save_dir=f'./{TODAY}',
                               tags=[selected_model, str(config["img_size"])],
                               log_model=not DEV_MODE,
                               experiment=experiment,
                               offline=DEV_MODE)

    # checkpoint
    checkpoint = ModelCheckpoint(
        dirpath=DIR,
        filename=f'{selected_model}-' +
        f'subset={config["subset"]}-'
        f'img_size={config["img_size"]}-' +
        f'fold={FOLD}-' +
        '{train_acc:.5f}-' +
        '{val_acc:.5f}-' +
        '{epoch: 02d}-',
        monitor='val_acc',
        mode='max')

    # early stopping
    es = EarlyStopping(monitor='val_acc', patience=3, mode='max')

    # modelModule
    cassava_model = LitModelClass(selected_model,
                                  num_classes=NUM_CLASSES,
                                  config=config)

    # dataModule
    cassava_data = LitDataClass(
        config=config,
        fold=config['fold'],
        subset=config['subset'],
        batch_size=config['batch_size'],
        train_val_split=0.9,
        use_extra_data=config['use_extra_data'])

    trainer = pl.Trainer(deterministic=True,
                         gpus=-1,
                         progress_bar_refresh_rate=25,
                         max_epochs=EPOCHS,
                         callbacks=[es, checkpoint],
                         logger=wandb_logger,
                         fast_dev_run=DEV_MODE)

    trainer.fit(model=cassava_model,
                datamodule=cassava_data)
