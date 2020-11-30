from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
import numpy as np
from params import NUM_CLASSES
import torch

ONE_HOT = np.eye(NUM_CLASSES)


def loss_fn(pred: torch.Tensor, truth: torch.Tensor):
    """Function to calculate the loss

    Args:
        pred (torch.Tensor): Predicted probabilities
        truth (torch.Tensor): Ground truths

    Returns:
        loss_value: Computed cross-entropy loss
    """

    if truth is None or pred is None:
        return None

    return CrossEntropyLoss()(pred, truth)


def accuracy_fn(pred: torch.Tensor, truth: torch.Tensor):
    """Function to calculate the accuracy

    Args:
        pred (torch.Tensor): Predicted probabilities
        truth (torch.Tensor): Ground truths

    Returns:
        accuracy_value: Computed accuracy
    """

    if truth is None or pred is None:
        return None

    return pl.metrics.Accuracy()(pred, truth)
