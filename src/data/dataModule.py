from params import SPLIT_SIZE, BATCH_SIZE, NUM_WORKERS, DATA_PATH
from pytorch_lightning import LightningDataModule
import pandas as pd
from sklearn import model_selection
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data.dataset import CassavaDataset

# TODO : to complete


class LitDataClass(LightningDataModule):
    def __init__(self,
                 batch_size: int = BATCH_SIZE,
                 train_val_split: float = SPLIT_SIZE):
        super().__init__(LitDataClass, self)
        self.val_data = None
        self.test_data = None
        self.train_data = None
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # just download the data
        # since the data is locally
        # we ignore this step
        datasets.MNIST
        pass

    def setup(self, na=1):
        # reading the data into a csv
        train_csv = pd.read_csv(f"{DATA_PATH}/train.csv")

        # adding a column for image location
        train_csv['path'] = train_csv['image_id'].map(
            lambda x: f"{DATA_PATH}/train_images/{x}")

        # shuffling and reset index
        train_csv.drop('image_id', axis=1, inplace=True)
        train_csv = train_csv.sample(frac=0.3).reset_index(drop=True)

        # Stratified kFold cross validation
        kFold = model_selection.StratifiedKFold(n_splits=5)

        for f, (t_, v_) in enumerate(kFold.split(X=train_csv,
                                                 y=train_csv.label)):
            train_csv.loc[v_, 'kFold'] = f

        # creating train and val dataframes based on SkFold
        self.train_data, self.val_data = [x for _, x in
                                          train_csv.groupby(train_csv['kFold'] == 3)]

        self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)
        self.val_data = self.val_data.sample(frac=1).reset_index(drop=True)

        self.train_dataset = CassavaDataset(self.train_data, {})
        self.val_dataset = CassavaDataset(self.val_data, {})

        # delete non useful constants
        del self.train_data
        del self.val_data

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS)
        return val_dataloader

    # def test_dataloader(self):
    #     test_dataloader = DataLoader(
    #         self.test_data,
    #         batch_size=self.batch_size,
    #         num_workers=NUM_WORKERS)
    #     return test_dataloader
