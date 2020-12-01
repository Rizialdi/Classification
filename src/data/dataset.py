from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CassavaDataset(Dataset):
    """Torch dataset for the problem

    Args:
        Dataset (Dataframe): Pandas dataframe containing informations
    """

    def __init__(self, df, params, train=True):
        self.df = df
        self.train = train
        self.params = params
        self.y = self.df['label']
        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx]['path']
        input_image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        return input_tensor, self.y[idx]
