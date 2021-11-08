import torch
import torch.nn.functional as F


class ConvGRU(torch.nn.Module):
    """A ConvGRU implementation."""

    def __init__(self, kernel_size = 3, sn_eps = 0.0001):
        """Constructor.

        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._sn_eps = sn_eps

    def forward(self, x, prev_state):
        """
        ConvGRU forward, returning the current+new state

        Args:
            x: Input tensor
            prev_state: Previous state

        Returns:
            New tensor plus the new state
        """
        # Concatenate the inputs and previous state along the channel axis.
        num_channels = prev_state.shape[1]
        xh = torch.cat([x, prev_state], dim=1)

        # Read gate of the GRU.
        read_gate_conv = layers.SNConv2D(
            num_channels, self._kernel_size, sn_eps=self._sn_eps)
        read_gate = F.sigmoid(read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate_conv = layers.SNConv2D(
            num_channels, self._kernel_size, sn_eps=self._sn_eps)
        update_gate = F.sigmoid(update_gate_conv(xh))

        # Gate the inputs.
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)

        # Gate the cell and state / outputs.
        output_conv = layers.SNConv2D(
            num_channels, self._kernel_size, sn_eps=self._sn_eps)
        c = F.relu(output_conv(gated_input))
        out = update_gate * prev_state + (1. - update_gate) * c
        new_state = out

        return out, new_state
