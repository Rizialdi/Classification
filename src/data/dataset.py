from PIL import Image
import torch
from torch.utils.data import Dataset
from data.transforms import simple_image_preprocess


class CassavaDataset(Dataset):
    """Torch dataset for the problem

    Args:
        Dataset (Dataframe): Pandas dataframe containing informations
    """

    def __init__(self, df, augmentations=None, train=True):
        self.df = df
        self.train = train
        self.augmentations = augmentations
        self.y = self.df['label']
        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx]['path']
        input_image = Image.open(image_path)

        if self.augmentations is not None:
            input_tensor = self.augmentations(image=input_image)
            input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)

        else:
            input_tensor = simple_image_preprocess(input_image)

        return input_tensor, self.y[idx]
