import torch
import torch.nn as nn
from torch.nn import functional as F
import einops


class SelfAttention2d(nn.Module):
    r"""Self Attention Module as proposed in the paper `"Self-Attention Generative Adversarial
    Networks by Han Zhang et. al." <https://arxiv.org/abs/1805.08318>`_
    .. math:: attention = softmax((query(x))^T * key(x))
    .. math:: output = \gamma * value(x) * attention + x
    where
    - :math:`query` : 2D Convolution Operation
    - :math:`key` : 2D Convolution Operation
    - :math:`value` : 2D Convolution Operation
    - :math:`x` : Input
    Args:
        input_dims (int): The input channel dimension in the input ``x``.
        output_dims (int, optional): The output channel dimension. If ``None`` the output
            channel value is computed as ``input_dims // 8``. So if the ``input_dims`` is **less
            than 8** then the layer will give an error.
        return_attn (bool, optional): Set it to ``True`` if you want the attention values to be
            returned.
    """

    def __init__(self, input_dims, output_dims=None, return_attn=False):
        output_dims = input_dims // 8 if output_dims is None else output_dims
        if output_dims == 0:
            raise Exception(
                "The output dims corresponding to the input dims is 0. Increase the input\
                            dims to 8 or more. Else specify output_dims"
            )
        super(SelfAttention2d, self).__init__()
        self.query = nn.Conv2d(input_dims, output_dims, 1)
        self.key = nn.Conv2d(input_dims, output_dims, 1)
        self.value = nn.Conv2d(input_dims, input_dims, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.return_attn = return_attn

    def forward(self, x):
        r"""Computes the output of the Self Attention Layer
        Args:
            x (torch.Tensor): A 4D Tensor with the channel dimension same as ``input_dims``.
        Returns:
            A tuple of the ``output`` and the ``attention`` if ``return_attn`` is set to ``True``
            else just the ``output`` tensor.
        """
        dims = (x.size(0), -1, x.size(2) * x.size(3))
        out_query = self.query(x).view(dims)
        out_key = self.key(x).view(dims).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(out_key, out_query), dim=-1)
        out_value = self.value(x).view(dims)
        out_value = torch.bmm(out_value, attn).view(x.size())
        out = self.gamma * out_value + x
        if self.return_attn:
            return out, attn
        return out


def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    k = einops.rearrange(k, "h w c -> (h w) c") # [h, w, c] -> [L, c]
    v = einops.rearrange(v, "h w c -> (h w) c") # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    beta = F.softmax(torch.einsum('hwc, Lc->hwL', q, k), dim=-1)

    # Einstein summation corresponding to the attention * value operation.
    out = torch.einsum('hwL, Lc->hwc', beta, v)
    return out


class AttentionLayer(torch.nn.Module):
    """Attention Module"""

    def __init__(self, input_channels: int, output_channels: int, ratio_kq = 8, ratio_v = 8):
        super(AttentionLayer, self).__init__()

        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.output_channels = output_channels
        self.input_channels = input_channels

        # Compute query, key and value using 1x1 convolutions.
        self.query = torch.nn.Conv2d(in_channels = input_channels,
            out_channels=self.output_channels // self.ratio_kq,
            kernel_size=(1,1), padding='VALID', bias=False)
        self.key = torch.nn.Conv2d(in_channels = input_channels,
                                   out_channels=self.output_channels // self.ratio_kq,
                                   kernel_size=(1,1), padding='VALID', bias=False)
        self.value = torch.nn.Conv2d(in_channels = input_channels,
                                     out_channels=self.output_channels // self.ratio_v,
                                     kernel_size=(1,1), padding='VALID', bias=False)

        self.last_conv = torch.nn.Conv2d(in_channels = self.output_channels,
                                         out_channels=self.output_channels,
                                         kernel_size=(1,1), padding='VALID', bias=False)

        # Learnable gain parameter
        self.gamma = torch.zeros(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute query, key and value using 1x1 convolutions.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Apply the attention operation.
        # TODO See can speed this up, ApplyAlongAxis isn't defined in the pseudocode
        out = []
        for b in range(x.shape[0]):
            # Apply to each in batch
            out.append(attention_einsum(query[b], key[b], value[b]))
        out = torch.stack(out, dim = 0)
        out = self._gamma * self.last_conv(out)

        # Residual connection.
        return out + x
