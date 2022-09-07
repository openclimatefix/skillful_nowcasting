""" Test loss functions"""
from dgmr.losses import SSIMLoss, MS_SSIMLoss
import torch


def test_ssim_loss():

    x = torch.rand((2, 3, 32, 32))
    y = torch.rand((2, 3, 32, 32))

    loss = SSIMLoss()
    assert float(loss(x=x, y=x)) == 0
    assert float(loss(x=x, y=y)) != 0

    loss = SSIMLoss(convert_range=True)
    assert float(loss(x=x, y=y)) != 0


def test_ms_ssim_loss():

    x = torch.rand((2, 3, 256, 256))
    y = torch.rand((2, 3, 256, 256))

    loss = MS_SSIMLoss()
    assert float(loss(x=x, y=x)) == 0
    assert float(loss(x=x, y=y)) != 0

    loss = MS_SSIMLoss(convert_range=True)
    assert float(loss(x=x, y=y)) != 0
