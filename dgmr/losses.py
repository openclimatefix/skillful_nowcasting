"""Module for various loss functions."""
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM, SSIM
from torch.nn import functional as F


class SSIMLoss(nn.Module):
    """SSIM loss class."""

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        SSIM Loss, optionally converting input range from [-1,1] to [0,1].

        Args:
            convert_range: decides whether to normalize the input range
            **kwargs: configurations values for the MS_SSIM instance.
        """
        super(SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Compute the scale structural similarity between two tensors."""
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class MS_SSIMLoss(nn.Module):
    """Multi-Scale SSIM loss class."""

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        Multi-Scale SSIM Loss, optionally converting input range from [-1,1] to [0,1].

        Args:
            convert_range: decides whether to normalize the input range
            **kwargs: configurations values for the MS_SSIM instance.
        """
        super(MS_SSIMLoss, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Compute the multi scale structural similarity between two tensors."""
        if self.convert_range:
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        return 1.0 - self.ssim_module(x, y)


class SSIMLossDynamic(nn.Module):
    """SSIM loss dynamic class."""

    def __init__(self, convert_range: bool = False, **kwargs):
        """
        SSIM loss dynamic initialize function.

        SSIM Loss on only dynamic part of the images, optionally converting input range
        from [-1,1] to [0,1].

        In Mathieu et al. to stop SSIM regressing towards the mean and predicting only the
        background, they only run SSIM on the dynamic parts of the image. We can accomplish
        that by subtracting the current image from the future ones.

        Args:
            convert_range: decides whether to normalize the input range
            **kwargs: configurations values for the MS_SSIM instance.
        """
        super(SSIMLossDynamic, self).__init__()
        self.convert_range = convert_range
        self.ssim_module = MS_SSIM(**kwargs)

    def forward(self, curr_image: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """Compute Structural Similarity (SSIM) loss from dynamic parts of the image."""
        if self.convert_range:
            curr_image = torch.div(torch.add(curr_image, 1), 2)
            x = torch.div(torch.add(x, 1), 2)
            y = torch.div(torch.add(y, 1), 2)
        # Subtract 'now' image to get what changes for both x and y
        x = x - curr_image
        y = y - curr_image
        # TODO: Mask out loss from pixels that don't change
        return 1.0 - self.ssim_module(x, y)


def tv_loss(img, tv_weight):
    """
    Taken from https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py.

    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class TotalVariationLoss(nn.Module):
    """Total variation loss class."""

    def __init__(self, tv_weight: float = 1.0):
        """
        Initialize the tv weight.

        Args:
            tv_weight: total variation weight (float)
        """
        super(TotalVariationLoss, self).__init__()
        self.tv_weight = tv_weight

    def forward(self, x: torch.Tensor):
        """Compute the total variation loss."""
        return tv_loss(x, self.tv_weight)


class GradientDifferenceLoss(nn.Module):
    """GradientDifferenceLoss class."""

    def __init__(self, alpha: int = 2):
        """Initialize the gradient difference loss class."""
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Compute the gradient loss difference between two tensors."""
        t1 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
                - torch.abs(y[:, :, :, 1:, :] - y[:, :, :, :-1, :])
            ),
            self.alpha,
        )
        t2 = torch.pow(
            torch.abs(
                torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:])
                - torch.abs(y[:, :, :, :, :-1] - y[:, :, :, :, 1:])
            ),
            self.alpha,
        )
        loss = t1 + t2
        print(loss.shape)
        return loss


class GridCellLoss(nn.Module):
    """Grid Cell Regularizer loss from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf."""

    def __init__(self, weight_fn=None, precip_weight_cap=24.0):
        """
        Initialize GridCellLoss.

        Args:
            weight_fn: A function to compute weights for the loss.
            precip_weight_cap: Custom ceiling value for the weight function.
        """
        super().__init__()
        self.weight_fn = lambda y: weight_fn(y, precip_weight_cap) if weight_fn else None

    def forward(self, generated_images, targets):
        """
        Forward function.

        Calculates the grid cell regularizer value, assumes generated images are the mean
        predictions from 6 calls to the generater (Monte Carlo estimation of the
        expectations for the latent variable)

        Args:
            generated_images: Mean generated images from the generator
            targets: Ground truth future frames

        Returns:
            Grid Cell Regularizer term
        """
        difference = generated_images - targets
        if self.weight_fn is not None:
            weights = self.weight_fn(targets)
            difference = difference * weights
        difference = difference.norm(p=1)
        return difference / targets.size(1) * targets.size(3) * targets.size(4)


class NowcastingLoss(nn.Module):
    """Nowcast Loss class."""

    def __init__(self):
        """Initialize function."""
        super().__init__()

    def forward(self, x, real_flag):
        """Apply the relu function to the input tensor."""
        if real_flag is True:
            x = -x
        return F.relu(1.0 + x).mean()


class FocalLoss(nn.Module):
    """
    Focal loss class.

    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported
    which is proposed in'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples
    (p>0.5) putting more focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are
    averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma: int = 2,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ):
        """Initialize Focal lost."""
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        """Forward function."""
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def loss_hinge_disc(score_generated, score_real):
    """Discriminator hinge loss."""
    l1 = F.relu(1.0 - score_real)
    loss = torch.mean(l1)
    l2 = F.relu(1.0 + score_generated)
    loss += torch.mean(l2)
    return loss


def loss_hinge_gen(score_generated):
    """Generate hinge loss."""
    loss = -torch.mean(score_generated)
    return loss


def grid_cell_regularizer(generated_samples, batch_targets):
    """Grid cell regularizer.

    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

    Returns:
      loss: A tensor of shape [batch_size].
    """
    gen_mean = torch.mean(generated_samples, dim=0)
    # TODO Possibly change clip here?
    weights = torch.clip(batch_targets, 0.0, 24.0)
    loss = torch.mean(torch.abs(gen_mean - batch_targets) * weights)
    return loss


def get_loss(loss: str = "mse", **kwargs) -> torch.nn.Module:
    """Return a loss type based on the passed in string."""
    if isinstance(loss, torch.nn.Module):
        return loss
    assert loss in [
        "mse",
        "bce",
        "binary_crossentropy",
        "crossentropy",
        "focal",
        "ssim",
        "ms_ssim",
        "l1",
        "tv",
        "total_variation",
        "ssim_dynamic",
        "gdl",
        "gradient_difference_loss",
    ]
    if loss == "mse":
        criterion = F.mse_loss
    elif loss in ["bce", "binary_crossentropy", "crossentropy"]:
        criterion = F.nll_loss
    elif loss in ["focal"]:
        criterion = FocalLoss()
    elif loss in ["ssim"]:
        criterion = SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ms_ssim"]:
        criterion = MS_SSIMLoss(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["ssim_dynamic"]:
        criterion = SSIMLossDynamic(data_range=1.0, size_average=True, **kwargs)
    elif loss in ["l1"]:
        criterion = torch.nn.L1Loss()
    elif loss in ["tv", "total_variation"]:
        criterion = TotalVariationLoss(tv_weight=kwargs.get("tv_weight", 1))
    elif loss in ["gdl", "gradient_difference_loss"]:
        criterion = GradientDifferenceLoss(alpha=kwargs.get("alpha", 2))
    else:
        raise ValueError(f"loss {loss} not recognized")
    return criterion
