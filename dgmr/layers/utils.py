"""Utilities functions for the layer submodules."""
import torch

from dgmr.layers import CoordConv


def get_conv_layer(conv_type: str = "standard") -> torch.nn.Module:
    """Return a conv layer based on the passed in string name."""
    if conv_type == "standard":
        conv_layer = torch.nn.Conv2d
    elif conv_type == "coord":
        conv_layer = CoordConv
    elif conv_type == "3d":
        conv_layer = torch.nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer
