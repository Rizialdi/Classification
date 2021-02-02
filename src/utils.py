from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
import numpy as np
from params import NUM_CLASSES
import requests
from decouple import config
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


def telegram_bot(bot_message):
    # load env variables from .env
    #bot_token = config('BOT_TOKEN')
    #bot_chatID = config('CHAT_ID')

    #send_text = \
    #   f"https://api.telegram.org/bot{bot_token}/" + \
    #   f"sendMessage?chat_id={bot_chatID}" + \
    #   f"&parse_mode=Markdown&text={bot_message}"

    #requests.post(send_text)
    pass
