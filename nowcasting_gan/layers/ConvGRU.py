import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from typing import Union, List, Tuple

# ------------------------------------------------------------------------------
# One-dimensional Convolution Gated Recurrent Unit
# ------------------------------------------------------------------------------


class ConvGRU1DCell(nn.Module):

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            recurrent_kernel_size: int = 3,
    ):
        """
        One-Dimensional Convolutional Gated Recurrent Unit (ConvGRU1D) cell.

        The input-to-hidden convolution kernel can be defined arbitrarily using
        the kernel_size, stride and padding parameters. The hidden-to-hidden
        convolution kernel is forced to be unit-stride, with a padding assuming
        an odd kernel size, in order to keep the number of features the same.

        The hidden state is initialized by default to a zero tensor of the
        appropriate shape.

        Arguments:
            input_channels {int} -- [Number of channels of the input tensor]
            hidden_channels {int} -- [Number of channels of the hidden state]
            kernel_size {int} -- [Size of the input-to-hidden convolving kernel]

        Keyword Arguments:
            stride {int} -- [Stride of the input-to-hidden convolution]
                             (default: {1})
            padding {int} -- [Zero-padding added to both sides of the input]
                              (default: {0})
            recurrent_kernel_size {int} -- [Size of the hidden-to-hidden
                                            convolving kernel] (default: {3})
        """
        super(ConvGRU1DCell, self).__init__()
        # ----------------------------------------------------------------------
        self.kernel_size = kernel_size
        self.stride = stride
        self.h_channels = hidden_channels
        self.padding_ih = padding
        self.padding_hh = recurrent_kernel_size // 2
        # ----------------------------------------------------------------------
        self.weight_ih = nn.Parameter(
            torch.ones(hidden_channels * 3, input_channels, kernel_size),
            requires_grad=True,
        )
        self.weight_hh = nn.Parameter(
            torch.ones(hidden_channels * 3, input_channels, recurrent_kernel_size),
            requires_grad=True,
        )
        self.bias_ih = nn.Parameter(torch.zeros(hidden_channels * 3), requires_grad=True)
        self.bias_hh = nn.Parameter(torch.zeros(hidden_channels * 3), requires_grad=True)
        # ----------------------------------------------------------------------
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_hh)
        init.xavier_uniform_(self.weight_ih)
        init.zeros_(self.bias_hh)
        init.zeros_(self.bias_ih)

    # --------------------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------------------

    def forward(self, input, hx=None):
        output_size = (
                int((input.size(-1) - self.kernel_size + 2 * self.padding_ih) / self.stride) + 1
        )
        # Handle the case of no hidden state provided
        if hx is None:
            hx = torch.zeros(input.size(0), self.h_channels, output_size, device=input.device)
        # Run the optimized convgru-cell
        return _opt_convgrucell_1d(
            input,
            hx,
            self.h_channels,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.stride,
            self.padding_ih,
            self.padding_hh,
        )


# ------------------------------------------------------------------------------
# Two-dimensional Convolution Gated Recurrent Unit
# ------------------------------------------------------------------------------


class ConvGRU2DCell(nn.Module):

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = (1, 1),
            padding: Union[int, Tuple[int, int]] = (0, 0),
            recurrent_kernel_size: Union[int, Tuple[int, int]] = (3, 3),
    ):
        """
        Two-Dimensional Convolutional Gated Recurrent Unit (ConvGRU2D) cell.

        The input-to-hidden convolution kernel can be defined arbitrarily using
        the kernel_size, stride and padding parameters. The hidden-to-hidden
        convolution kernel is forced to be unit-stride, with a padding assuming
        an odd kernel size in both dimensions, in order to keep the number of
        features the same.

        The hidden state is initialized by default to a zero tensor of the
        appropriate shape.

        Arguments:
            input_channels {int} -- [Number of channels of the input tensor]
            hidden_channels {int} -- [Number of channels of the hidden state]
            kernel_size {int or tuple} -- [Size of the input-to-hidden
                                           convolving kernel]

        Keyword Arguments:
            stride {int or tuple} -- [Stride of the input-to-hidden convolution]
                                      (default: {(1, 1)})
            padding {int or tuple} -- [Zero-padding added to both sides of the
                                       input] (default: {0})
            recurrent_kernel_size {int or tuple} -- [Size of the hidden-to-
                                                     -hidden convolving kernel]
                                                     (default: {(3, 3)})
        """
        super(ConvGRU2DCell, self).__init__()
        # ----------------------------------------------------------------------
        # Handle int to tuple conversion
        if isinstance(recurrent_kernel_size, int):
            recurrent_kernel_size = (recurrent_kernel_size,) * 2
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(padding, int):
            padding = (padding,) * 2
        # ----------------------------------------------------------------------
        # Save input parameters for later
        self.kernel_size = kernel_size
        self.stride = stride
        self.h_channels = hidden_channels
        self.padding_ih = padding
        self.padding_hh = (
            recurrent_kernel_size[0] // 2,
            recurrent_kernel_size[1] // 2,
        )
        # ----------------------------------------------------------------------
        # Initialize the convolution kernels
        self.weight_ih = nn.Parameter(
            torch.ones(
                hidden_channels * 3,
                input_channels,
                kernel_size[0],
                kernel_size[1],
                ),
            requires_grad=True,
        )
        self.weight_hh = nn.Parameter(
            torch.ones(
                hidden_channels * 3,
                input_channels,
                recurrent_kernel_size[0],
                recurrent_kernel_size[1],
                ),
            requires_grad=True,
        )
        self.bias_ih = nn.Parameter(torch.zeros(hidden_channels * 3), requires_grad=True)
        self.bias_hh = nn.Parameter(torch.zeros(hidden_channels * 3), requires_grad=True)
        # ----------------------------------------------------------------------
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_hh)
        init.xavier_uniform_(self.weight_ih)
        init.zeros_(self.bias_hh)
        init.zeros_(self.bias_ih)

    # --------------------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------------------

    def forward(self, input, hx=None):
        output_size = (
            int((input.size(-2) - self.kernel_size[0] + 2 * self.padding_ih[0]) / self.stride[0])
            + 1,
            int((input.size(-1) - self.kernel_size[1] + 2 * self.padding_ih[1]) / self.stride[1])
            + 1,
        )
        # Handle the case of no hidden state provided
        if hx is None:
            hx = torch.zeros(input.size(0), self.h_channels, *output_size, device=input.device)
        # Run the optimized convgru-cell
        return _opt_convgrucell_2d(
            input,
            hx,
            self.h_channels,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.stride,
            self.padding_ih,
            self.padding_hh,
        )


class ConvGRU(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = (1, 1),
            padding: Union[int, Tuple[int, int]] = (0, 0),
            recurrent_kernel_size: Union[int, Tuple[int, int]] = (3, 3),
    ):
        """
        Recurrent wrapper for use with any rnn cell that takes as input a tensor
        and a hidden state and returns an updated hidden state. This wrapper
        returns the full sequence of hidden states. It assumes the first
        dimension corresponds to the timesteps, and that the other dimensions
        are directly compatible with the given rnn cell.
        Implements a very basic truncated backpropagation through time
        corresponding to the case k1=k2 (see 'An Efficient Gradient-Based
        Algorithm for On-Line Training of Recurrent Network Trajectories',
        Ronald J. Williams and Jing Pen, Neural Computation, vol. 2,
        pp. 490-501, 1990).
        Args:
            rnn_cell (nn.Module): [The torch module that takes one timestep of
                the input tensor and the hidden state and returns a new hidden
                state]
            truncation_steps (int, optional): [The maximum number of timesteps
                to include in the backpropagation graph. This can help speed up
                runtime on CPU and avoid vanishing gradient problems, however
                it is mostly useful for very long sequences]. Defaults to None.
        """
        super(ConvGRU, self).__init__()

        self.rnn_cell = ConvGRU2DCell(
            input_channels, hidden_channels, kernel_size, stride, padding, recurrent_kernel_size
        )

    def forward(self, input, hidden_state=None):
        output = []
        for step in range(input.size(1)):
            # Compute current time-step
            hidden_state = self.rnn_cell(input[:, step, :, :, :], hidden_state)
            output.append(hidden_state)
        # Stack the list of output hidden states into a tensor
        output = torch.stack(output, 0)
        return output


# --------------------------------------------------------------------------
# Torchscript optimized cell functions
# --------------------------------------------------------------------------


@torch.jit.script
def _opt_cell_end(hidden, ih_1, hh_1, ih_2, hh_2, ih_3, hh_3):
    z = torch.sigmoid(ih_1 + hh_1)
    r = torch.sigmoid(ih_2 + hh_2)
    n = torch.tanh(ih_3 + r * hh_3)
    out = (1 - z) * n + z * hidden
    return out


@torch.jit.script
def _opt_convgrucell_1d(
        inputs,
        hidden,
        channels: int,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        stride: int,
        pad1: int,
        pad2: int,
):
    ih_output = functional.conv1d(inputs, w_ih, bias=b_ih, stride=stride, padding=pad1)
    hh_output = functional.conv1d(hidden, w_hh, bias=b_hh, stride=1, padding=pad2)
    output = _opt_cell_end(
        hidden,
        torch.narrow(ih_output, 1, 0, channels),
        torch.narrow(hh_output, 1, 0, channels),
        torch.narrow(ih_output, 1, channels, channels),
        torch.narrow(hh_output, 1, channels, channels),
        torch.narrow(ih_output, 1, 2 * channels, channels),
        torch.narrow(hh_output, 1, 2 * channels, channels),
    )
    return output


@torch.jit.script
def _opt_convgrucell_2d(
        inputs,
        hidden,
        channels: int,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        stride: List[int],
        pad1: List[int],
        pad2: List[int],
):
    ih_output = functional.conv2d(inputs, w_ih, bias=b_ih, stride=stride, padding=pad1)
    hh_output = functional.conv2d(hidden, w_hh, bias=b_hh, stride=1, padding=pad2)
    output = _opt_cell_end(
        hidden,
        torch.narrow(ih_output, 1, 0, channels),
        torch.narrow(hh_output, 1, 0, channels),
        torch.narrow(ih_output, 1, channels, channels),
        torch.narrow(hh_output, 1, channels, channels),
        torch.narrow(ih_output, 1, 2 * channels, channels),
        torch.narrow(hh_output, 1, 2 * channels, channels),
    )
    return output
