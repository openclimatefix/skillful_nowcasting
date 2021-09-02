import torch
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from torch.nn.utils import spectral_norm
from torchvision.transforms import RandomCrop
from nowcasting_gan.common import DBlock


class NowcastingTemporalDiscriminator(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 12,
            crop_size: int = 128,
            num_layers: int = 3,
            conv_type: str = "standard",
    ):
        """
        Temporal Discriminator from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of channels per timestep
            crop_size: Size of the crop, in the paper half the width of the input images
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        self.transform = RandomCrop(crop_size)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 48
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=internal_chn * input_channels,
            conv_type="3d",
            first_relu=False,
        )
        self.d2 = DBlock(
            input_channels=internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            conv_type="3d",
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )

        self.d_last = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        x = self.space2depth(x)
        # Have to move time and channels
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        x = self.d1(x)
        x = self.d2(x)
        # Convert back to T x C x H x W
        x = torch.permute(x, dims=(0, 2, 1, 3, 4))
        # Per Timestep part now, same as spatial discriminator
        representations = []
        for idx in range(x.size(1)):
            # Intermediate DBlocks
            rep = x[:, idx, :, :, :]
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d_last(rep)
            # Sum-pool along width and height all 8 representations, pretty sure only the last output
            rep = torch.sum(rep.view(rep.size(0), rep.size(1), -1), dim=2)
            # rep = self.fc(rep)
            representations.append(rep)
        # The representations are summed together before the ReLU
        x = torch.stack(representations, dim=0).sum(dim=0)  # Should be right shape? TODO Check
        # ReLU the output
        x = self.fc(x)
        # x = self.relu(x)
        return x


class NowcastingSpatialDiscriminator(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 12,
            num_timesteps: int = 8,
            num_layers: int = 4,
            conv_type: str = "standard",
    ):
        """
        Spatial discriminator from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            num_timesteps: Number of timesteps to use, in the paper 8/18 timesteps were chosen
            num_layers: Number of intermediate DBlock layers to use
            conv_type: Type of 2d convolutions to use, see satflow/models/utils.py for options
        """
        super().__init__()
        # Randomly, uniformly, select 8 timesteps to do this on from the input
        self.num_timesteps = num_timesteps
        # First step is mean pooling 2x2 to reduce input by half
        self.mean_pool = torch.nn.AvgPool2d(2)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        internal_chn = 24
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=2 * internal_chn * input_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            internal_chn *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    input_channels=internal_chn * input_channels,
                    output_channels=2 * internal_chn * input_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            input_channels=2 * internal_chn * input_channels,
            output_channels=2 * internal_chn * input_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        # Spectrally normalized linear layer for binary classification
        self.fc = spectral_norm(torch.nn.Linear(2 * internal_chn * input_channels, 1))
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x should be the chosen 8 or so
        idxs = torch.randint(low=0, high=x.size()[1], size=(self.num_timesteps,))
        representations = []
        for idx in idxs:
            rep = self.mean_pool(x[:, idx, :, :, :])  # 128x128
            rep = self.space2depth(rep)  # 64x64x4
            rep = self.d1(rep)  # 32x32
            # Intermediate DBlocks
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d6(rep)  # 2x2

            # Sum-pool along width and height all 8 representations, pretty sure only the last output
            rep = torch.sum(rep.view(rep.size(0), rep.size(1), -1), dim=2)
            representations.append(rep)

        # The representations are summed together before the ReLU
        x = torch.stack(representations, dim=0).sum(dim=0)  # Should be right shape? TODO Check
        # ReLU the output
        x = self.fc(x)
        # x = self.relu(x)
        return x
