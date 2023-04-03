""" Test loss functions"""
from dgmr.losses import SSIMLoss, MS_SSIMLoss, SSIMLossDynamic, tv_loss
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


def test_ssim_loss_dynamic():
    x = torch.rand((2, 3, 256, 256))
    y = torch.rand((2, 3, 256, 256))
    curr_image = torch.rand((2, 3, 256, 256))

    loss = SSIMLossDynamic()
    assert float(loss(x=x, y=x, curr_image=curr_image)) == 0
    assert float(loss(x=x, y=y, curr_image=curr_image)) != 0

    loss = SSIMLossDynamic(convert_range=True)
    assert float(loss(x=x, y=y, curr_image=curr_image)) != 0


def test_tv_loss():
    x = torch.ones((2, 3, 256, 256))
    x[0, 0, 0, 0] = 2.5

    assert float(tv_loss(img=x, tv_weight=2)) == 2 * (1.5**2 + 1.5**2)
