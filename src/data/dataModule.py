from params import SPLIT_SIZE, BATCH_SIZE, NUM_WORKERS, DATA_PATH, SEED
from pytorch_lightning import LightningDataModule
import pandas as pd
import os
from sklearn import model_selection
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import CassavaDataset

# TODO : to complete


class LitDataClass(LightningDataModule):
    def __init__(self,
                 fold: int = 0,
                 subset: float = 1.0,
                 batch_size: int = BATCH_SIZE,
                 train_val_split: float = SPLIT_SIZE,
                 use_extra_data: bool = False):
        super().__init__(LitDataClass, self)
        self.fold = fold
        self.subset = subset
        self.val_data = None
        self.test_data = None
        self.train_data = None
        self.batch_size = batch_size
        self.use_extra_data = use_extra_data
        self.train_val_split = train_val_split
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        # just download the data
        # since the data is locally
        # we ignore this step
        pass

    def setup(self, na=1):
        # defining folders
        folder_current = 'cassava-leaf-disease-classification'
        folder_old = 'cassava-disease'

        # reading the data into a csv
        train_csv = pd.read_csv(
            f"{DATA_PATH}/{folder_current}/train.csv")

        # adding a column for image location
        train_csv['path'] = train_csv['image_id'].map(
            lambda x: f"{DATA_PATH}/{folder_current}/train_images/{x}")

        # shuffling and reset index
        train_csv.drop('image_id', axis=1, inplace=True)

        # add extra data if use_extra_data = True
        if self.use_extra_data:
            olddir = f'{DATA_PATH}/{folder_old}/train'

            folder_to_label_mapper = {
                "cbb": 0,
                'cbsd': 1,
                'cgm': 2,
                'cmd': 3,
                'healthy': 4
            }

            paths = []
            labels = []
            for label in os.listdir(f'{olddir}'):
                pths = [
                    f'{olddir}/{label}/{x}'
                    for x in os.listdir(f'{olddir}/{label}')]
                labels += [folder_to_label_mapper[label]]*len(pths)
                paths += pths

            dico = {'label': labels, 'path': paths}
            train_extra_data = pd.DataFrame(data=dico)

            # append extra data to train_csv
            train_csv = train_csv.append(train_extra_data,
                                         ignore_index=True,
                                         verify_integrity=True)

            train_csv = train_csv.sample(frac=1).reset_index(drop=True)

        if self.subset:
            train_csv = train_csv.groupby('label').apply(
                lambda x: x.sample(frac=self.subset)).reset_index(drop=True)

        # Unbalanced dataset
        # shuffle/random_state -> same split over runs
        kFold = model_selection.StratifiedKFold(n_splits=5,
                                                shuffle=True,
                                                random_state=SEED)

        for f, (t_, v_) in enumerate(kFold.split(X=train_csv,
                                                 y=train_csv.label)):
            train_csv.loc[v_, 'kFold'] = f

        # creating train and val dataframes based on SkFold
        self.train_data = train_csv[train_csv['kFold'] == self.fold]
        self.val_data = train_csv[train_csv['kFold'] != self.fold]

        self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)
        self.val_data = self.val_data.sample(frac=1).reset_index(drop=True)

        self.train_dataset = CassavaDataset(
            self.train_data)
        self.val_dataset = CassavaDataset(
            self.val_data)

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
