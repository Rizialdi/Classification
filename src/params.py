import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

LR = 1e-3
SEED: int = 42
EPOCHS: int = 20
IMAGE_SIZE: int = 512
DROPOUT: float = 0.5
NUM_CLASSES: int = 5
NUM_WORKERS: int = 4
DEV_MODE: bool = False
BATCH_SIZE: int = 10
SUBSET: float = 1
SPLIT_SIZE: float = 0.9
FOLD: int = 4

DATA_PATH: str = "../../../input"
