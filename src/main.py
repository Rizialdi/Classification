from data.dataModule import LitDataClass
from params import NUM_CLASSES, BATCH_SIZE, EPOCHS
from training.train import LitModelClass
import pytorch_lightning as pl


if __name__ == "__main__":
    # model
    selected_model = "resnet18"
    cassava_model = LitModelClass(selected_model, num_classes=NUM_CLASSES)

    # data
    cassava_data = LitDataClass(batch_size=BATCH_SIZE, train_val_split=0.9)

    # fast_dev_run=True
    trainer = pl.Trainer(deterministic=True,
                         max_epochs=EPOCHS, fast_dev_run=True)

    trainer.fit(cassava_model, cassava_data)

    # Checkpoints folder

    # TODAY = str(datetime.date.today())
    # CP_TODAY = f"../checkpoints/{TODAY}/"

    # if not os.path.exists(CP_TODAY):
    #     os.mkdir(CP_TODAY)

    #     # Logger
