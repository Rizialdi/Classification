import torch
import torch.nn as nn
import resnest.torch as resnest_torch
from efficientnet_pytorch import EfficientNet


def get_model(name: str = 'resnet18', num_classes: int = 5) -> nn.Module:
    """
    Loads a pretrained model.
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    if "resnest" in name:
        model = getattr(resnest_torch, name)(pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif "resnext" in name or "resnet" in name:
        # resnet18 ... {34, 50, 1001, 152}
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif "efficientnet" in name:
        # name : 'efficientnet-b0' ... '-b7'
        model = EfficientNet.from_pretrained(name)
    else:
        raise NotImplementedError

    if "efficientnet" not in name and "se" not in name:
        nb_ft = model.fc.in_features
        del model.fc
        model.fc = nn.Linear(nb_ft, num_classes)
    else:
        nb_ft = model._fc.in_features
        del model._fc
        model._fc = nn.Linear(nb_ft, num_classes)

    return model


def count_parameters(model: nn.Module, all: bool = False):
    """
    Count the parameters of a model

    Arguments:
        model {torch module} -- Model to count the parameters of

    Keyword Arguments:
        all {bool} -- Whether to include not trainable parameters
        in the sum (default: {False})

    Returns:
        int -- Number of parameters
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
