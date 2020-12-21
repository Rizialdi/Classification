import albumentations as A
import numpy as np
from torchvision import transforms


def train_image_augmentation(image, img_size):
    image = np.array(image)

    augmentation = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.CenterCrop(img_size, img_size, p=1.0),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Blur(blur_limit=3),
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2,
                                 sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.5
                                 ),

            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            )

        ],
        p=1.0,
    )

    augmented_image = augmentation(image=image)

    return augmented_image['image']


def valid_image_augmentation(image, img_size):
    image = np.array(image)

    augmentation = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.CenterCrop(img_size, img_size, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    augmented_image = augmentation(image=image)

    return augmented_image['image']


def simple_image_preprocess(image, img_size):
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(image)
