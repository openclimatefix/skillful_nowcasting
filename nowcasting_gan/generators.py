import torch
from torch.nn.modules.pixelshuffle import PixelShuffle
from torch.nn.utils import spectral_norm
from typing import List
from nowcasting_gan.common import GBlock
from nowcasting_gan.layers import ConvGRU


class NowcastingSampler(torch.nn.Module):
    def __init__(
            self,
            forecast_steps: int = 18,
            latent_channels: int = 768,
            context_channels: int = 384,
            output_channels: int = 1,
    ):
        """
        Sampler from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        The sampler takes the output from the Latent and Context conditioning stacks and
        creates one stack of ConvGRU layers per future timestep.
        Args:
            forecast_steps: Number of forecast steps
            latent_channels: Number of input channels to the lowest ConvGRU layer
        """
        super().__init__()
        self.forecast_steps = forecast_steps
        self.convGRU1 = ConvGRU(
            input_channels=latent_channels,
            hidden_channels=context_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.g1 = GBlock(input_channels=latent_channels, output_channels=latent_channels // 2)
        self.convGRU2 = ConvGRU(
            input_channels=latent_channels // 2,
            hidden_channels=context_channels // 2,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.g2 = GBlock(input_channels=latent_channels // 2, output_channels=latent_channels // 4)
        self.convGRU3 = ConvGRU(
            input_channels=latent_channels // 4,
            hidden_channels=context_channels // 4,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.g3 = GBlock(input_channels=latent_channels // 4, output_channels=latent_channels // 8)
        self.convGRU4 = ConvGRU(
            input_channels=latent_channels // 8,
            hidden_channels=context_channels // 8,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        self.g4 = GBlock(
            input_channels=latent_channels // 8, output_channels=latent_channels // 16
        )
        self.bn = torch.nn.BatchNorm2d(latent_channels // 16)
        self.relu = torch.nn.ReLU()
        self.conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=latent_channels // 16, out_channels=4 * output_channels, kernel_size=1
            )
        )

        self.depth2space = PixelShuffle(upscale_factor=2)

        # Now make copies of the entire stack, one for each future timestep
        stacks = torch.nn.ModuleDict()
        for i in range(forecast_steps):
            stacks[f"forecast_{i}"] = torch.nn.ModuleList(
                [
                    self.convGRU1,
                    self.g1,
                    self.convGRU2,
                    self.g2,
                    self.convGRU3,
                    self.g3,
                    self.convGRU4,
                    self.g4,
                    self.bn,
                    self.relu,
                    self.conv_1x1,
                    self.depth2space,
                ]
            )
        self.stacks = stacks

    def forward(
            self, conditioning_states: List[torch.Tensor], latent_dim: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the sampling from Skillful Nowcasting with GANs
        Args:
            conditioning_states: Outputs from the `ContextConditioningStack` with the 4 input states, ordered from largest to smallest spatially
            latent_dim: Output from `LatentConditioningStack` for input into the ConvGRUs

        Returns:
            forecast_steps-length output of images for future timesteps

        """
        # Iterate through each forecast step
        # Initialize with conditioning state for first one, output for second one
        forecasts = []
        init_states = list(
            conditioning_states
        )  # [torch.unsqueeze(c, dim=1) for c in conditioning_states]
        # Need to expand latent dim to the batch size
        latent_dim = torch.cat(init_states[0].size()[0] * [latent_dim])
        latent_dim = torch.unsqueeze(latent_dim, dim=1)
        for i in range(self.forecast_steps):
            # Start at lowest one and go up, conditioning states
            # ConvGRU1
            x = self.stacks[f"forecast_{i}"][0](latent_dim, hidden_state=init_states[3])
            # Update for next timestep
            init_states[3] = torch.squeeze(x, dim=0)
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            # GBlock1
            x = self.stacks[f"forecast_{i}"][1](x)
            # Expand to 5D input
            x = torch.unsqueeze(x, dim=1)
            # ConvGRU2
            x = self.stacks[f"forecast_{i}"][2](x, hidden_state=init_states[2])
            # Update for next timestep
            init_states[2] = torch.squeeze(x, dim=0)
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            # GBlock2
            x = self.stacks[f"forecast_{i}"][3](x)
            # Expand to 5D input
            x = torch.unsqueeze(x, dim=1)
            # ConvGRU3
            x = self.stacks[f"forecast_{i}"][4](x, hidden_state=init_states[1])
            # Update for next timestep
            init_states[1] = torch.squeeze(x, dim=0)
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            # GBlock3
            x = self.stacks[f"forecast_{i}"][5](x)
            # Expand to 5D input
            x = torch.unsqueeze(x, dim=1)
            # ConvGRU4
            x = self.stacks[f"forecast_{i}"][6](x, hidden_state=init_states[0])
            # Update for next timestep
            init_states[0] = torch.squeeze(x, dim=0)
            # Reduce to 4D input
            x = torch.squeeze(x, dim=0)
            # GBlock4
            x = self.stacks[f"forecast_{i}"][7](x)
            # BN
            x = self.stacks[f"forecast_{i}"][8](x)
            # ReLU
            x = self.stacks[f"forecast_{i}"][9](x)
            # Conv 1x1
            x = self.stacks[f"forecast_{i}"][10](x)
            # Depth2Space
            x = self.stacks[f"forecast_{i}"][11](x)
            forecasts.append(x)
        # Convert forecasts to a torch Tensor
        forecasts = torch.stack(forecasts, dim=1)
        return forecasts
