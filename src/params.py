import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


LR = 1e-3
SEED: int = 42
EPOCHS: int = 64
NUM_CLASSES: int = 5
NUM_WORKERS: int = 4
BATCH_SIZE: int = 64
SPLIT_SIZE: float = 0.9

DATA_PATH: str = "../input/cassava-leaf-disease-classification"