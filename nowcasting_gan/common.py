from typing import Tuple
import torch
from torch.distributions import uniform
from torch.nn.utils import spectral_norm
from torch.nn.modules.pixelshuffle import PixelUnshuffle
from torch.nn.functional import interpolate
from nowcasting_gan.layers.utils import get_conv_layer
from nowcasting_gan.layers import SelfAttention2d


class GBlock(torch.nn.Module):
    def __init__(
            self, input_channels: int = 12, output_channels: int = 12, conv_type: str = "standard"
    ):
        """
        G Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(input_channels)
        self.bn2 = torch.nn.BatchNorm2d(input_channels)
        self.relu = torch.nn.ReLU()
        # Upsample in the 1x1
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
        )
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        # Upsample 2D conv
        self.first_conv_3x3 = torch.nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.last_conv_3x3 = conv2d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1
        x1 = self.upsample(x)
        x1 = self.conv_1x1(x1)

        # Branch 2
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_conv_3x3(
            x2, output_size=(2 * x.size()[-2], 2 * x.size()[-1])
        )  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine
        x = x1 + x2
        return x


class DBlock(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 12,
            output_channels: int = 12,
            conv_type: str = "standard",
            first_relu: bool = True,
            keep_same_output: bool = False,
    ):
        """
        D and 3D Block from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Convolution type, see satflow/models/utils.py for options
            first_relu: Whether to have an ReLU before the first 3x3 convolution
            keep_same_output: Whether the output should have the same spatial dimensions as input, if False, downscales by 2
        """
        super().__init__()
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
        )
        self.first_conv_3x3 = conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1,
        )
        self.last_conv_3x3 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        if conv_type == "3d":
            # Need spectrally normalized convolutions
            self.conv_1x1 = spectral_norm(self.conv_1x1)
            self.first_conv_3x3 = spectral_norm(self.first_conv_3x3)
            self.last_conv_3x3 = spectral_norm(self.last_conv_3x3)
        # Downsample at end of 3x3
        self.relu = torch.nn.ReLU()
        # Concatenate to double final channels and keep reduced spatial extent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv_1x1(x)
        if not self.keep_same_output:
            x1 = interpolate(
                x1, mode="trilinear" if self.conv_type == "3d" else "bilinear", scale_factor=0.5
            )  # Downscale by half
        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)
        if not self.keep_same_output:
            x = interpolate(
                x, mode="trilinear" if self.conv_type == "3d" else "bilinear", scale_factor=0.5
            )  # Downscale by half
        x = x1 + x  # Sum the outputs should be half spatial and double channels
        return x


class LBlock(torch.nn.Module):
    def __init__(
            self, input_channels: int = 12, output_channels: int = 12, conv_type: str = "standard"
    ):
        """
        L-Block for increasing the number of channels in the input
         from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            conv_type: Which type of convolution desired, see satflow/models/utils.py for options
        """
        super().__init__()
        # Output size should be channel_out - channel_in
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=input_channels,
            out_channels=output_channels - input_channels,
            kernel_size=1,
        )

        self.first_conv_3x3 = conv2d(
            input_channels, out_channels=output_channels, kernel_size=3, padding=1, stride=1
        )
        self.relu = torch.nn.ReLU()
        self.last_conv_3x3 = conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )

    def forward(self, x) -> torch.Tensor:
        x1 = self.conv_1x1(x)

        x2 = self.first_conv_3x3(x)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        x = x2 + (torch.cat((x, x1), dim=1))
        return x


class ContextConditioningStack(torch.nn.Module):
    def __init__(
            self,
            input_channels: int = 1,
            output_channels: int = 768,
            num_context_steps: int = 4,
            conv_type: str = "standard",
    ):
        """
        Conditioning Stack using the context images from Skillful Nowcasting, , see https://arxiv.org/pdf/2104.00954.pdf

        Args:
            input_channels: Number of input channels per timestep
            output_channels: Number of output channels for the lowest block
            conv_type: Type of 2D convolution to use, see satflow/models/utils.py for options
        """
        super().__init__()
        conv2d = get_conv_layer(conv_type)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        # Process each observation processed separately with 4 downsample blocks
        # Concatenate across channel dimension, and for each output, 3x3 spectrally normalized convolution to reduce
        # number of channels by 2, followed by ReLU
        # TODO Not sure if a different block for each timestep, or same block used separately
        self.d1 = DBlock(
            input_channels=4 * input_channels,
            output_channels=((output_channels // 4) * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d2 = DBlock(
            input_channels=((output_channels // 4) * input_channels) // num_context_steps,
            output_channels=((output_channels // 2) * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d3 = DBlock(
            input_channels=((output_channels // 2) * input_channels) // num_context_steps,
            output_channels=(output_channels * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d4 = DBlock(
            input_channels=(output_channels * input_channels) // num_context_steps,
            output_channels=(output_channels * 2 * input_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.conv1 = spectral_norm(
            conv2d(
                in_channels=(output_channels // 4) * input_channels,
                out_channels=(output_channels // 8) * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.conv2 = spectral_norm(
            conv2d(
                in_channels=(output_channels // 2) * input_channels,
                out_channels=(output_channels // 4) * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.conv3 = spectral_norm(
            conv2d(
                in_channels=output_channels * input_channels,
                out_channels=(output_channels // 2) * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.conv4 = spectral_norm(
            conv2d(
                in_channels=output_channels * 2 * input_channels,
                out_channels=output_channels * input_channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.relu = torch.nn.ReLU()

    def forward(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Each timestep processed separately
        x = self.space2depth(x)
        steps = x.size(1)  # Number of timesteps
        scale_1 = []
        scale_2 = []
        scale_3 = []
        scale_4 = []
        for i in range(steps):
            s1 = self.d1(x[:, i, :, :, :])
            s2 = self.d2(s1)
            s3 = self.d3(s2)
            s4 = self.d4(s3)
            scale_1.append(s1)
            scale_2.append(s2)
            scale_3.append(s3)
            scale_4.append(s4)
        scale_1 = torch.cat(scale_1, dim=1)  # B, T, C, H, W and want along C dimension
        scale_2 = torch.cat(scale_2, dim=1)  # B, T, C, H, W and want along C dimension
        scale_3 = torch.cat(scale_3, dim=1)  # B, T, C, H, W and want along C dimension
        scale_4 = torch.cat(scale_4, dim=1)  # B, T, C, H, W and want along C dimension
        # TODO Figure out where extra channels come from, paper says concat outputs and divide channels by 2 gives 48,96,192,384 total, but this gives 8*4 = 32, 16*4 = 64
        scale_1 = self.relu(self.conv1(scale_1))
        scale_2 = self.relu(self.conv2(scale_2))
        scale_3 = self.relu(self.conv3(scale_3))
        scale_4 = self.relu(self.conv4(scale_4))

        return scale_1, scale_2, scale_3, scale_4


class LatentConditioningStack(torch.nn.Module):
    def __init__(
            self,
            shape: (int, int, int) = (8, 8, 8),
            output_channels: int = 768,
            use_attention: bool = True,
    ):
        """
        Latent conditioning stack from Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf
        Converst
        Args:
            shape: Shape of the latent space, Should be (H/32,W/32,x) of the final image shape
            output_channels: Number of output channels for the conditioning stack
            use_attention: Whether to have a self-attention block or not
        """
        super().__init__()
        self.shape = shape
        self.use_attention = use_attention
        self.distribution = uniform.Uniform(low=torch.Tensor([0.0]), high=torch.Tensor([1.0]))

        self.conv_3x3 = torch.nn.Conv2d(
            in_channels=shape[0], out_channels=shape[0], kernel_size=3, padding=1
        )
        self.l_block1 = LBlock(input_channels=shape[0], output_channels=output_channels // 32)
        self.l_block2 = LBlock(
            input_channels=output_channels // 32, output_channels=output_channels // 16
        )
        self.l_block3 = LBlock(
            input_channels=output_channels // 16, output_channels=output_channels // 4
        )
        if self.use_attention:
            self.att_block = SelfAttention2d(
                input_dims=output_channels // 4, output_dims=output_channels // 4
            )
        self.l_block4 = LBlock(
            input_channels=output_channels // 4, output_channels=output_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: tensor on the correct device, to move over the latent distribution

        Returns:

        """
        z = self.distribution.sample(self.shape)
        # Batch is at end for some reason, reshape
        z = torch.permute(z, (3, 0, 1, 2)).type_as(x)
        z = self.conv_3x3(z)
        z = self.l_block1(z)
        z = self.l_block2(z)
        z = self.l_block3(z)
        if self.use_attention:
            z = self.att_block(z)
        z = self.l_block4(z)
        return z
