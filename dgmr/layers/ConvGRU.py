import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ConvGRUCell(torch.nn.Module):
    """A ConvGRU implementation."""

    def __init__(self, input_channels: int, output_channels: int, kernel_size=3, sn_eps=0.0001):
        """Constructor.

        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._sn_eps = sn_eps
        self.read_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.update_gate_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )
        self.output_conv = spectral_norm(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=(kernel_size, kernel_size),
                padding=1,
            ),
            eps=sn_eps,
        )

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
        xh = torch.cat([x, prev_state], dim=1)

        # Read gate of the GRU.
        read_gate = F.sigmoid(self.read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate = F.sigmoid(self.update_gate_conv(xh))

        # Gate the inputs.
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)

        # Gate the cell and state / outputs.
        c = F.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        new_state = out

        return out, new_state


class ConvGRU(torch.nn.Module):
    """ConvGRU Cell wrapper to replace tf.static_rnn in TF implementation"""

    def __init__(
        self, input_channels: int, output_channels: int, kernel_size: int = 3, sn_eps=0.0001
    ):
        super().__init__()
        self.cell = ConvGRUCell(input_channels, output_channels, kernel_size, sn_eps)

    def forward(self, x: torch.Tensor, hidden_state=None) -> torch.Tensor:
        outputs = []
        for step in range(len(x)):
            # Compute current timestep
            output, hidden_state = self.cell(x[step], hidden_state)
            outputs.append(output)
        # Stack outputs to return as tensor
        outputs = torch.stack(outputs, dim=0)
        return outputs
