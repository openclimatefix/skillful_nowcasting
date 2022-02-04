import torch
from dgmr.losses import (
    NowcastingLoss,
    GridCellLoss,
    loss_hinge_disc,
    loss_hinge_gen,
    grid_cell_regularizer,
)
import pytorch_lightning as pl
import torchvision
from dgmr.common import LatentConditioningStack, ContextConditioningStack
from dgmr.generators import Sampler, Generator
from dgmr.discriminators import Discriminator
from dgmr.hub import NowcastingModelHubMixin


class DGMR(pl.LightningModule, NowcastingModelHubMixin):
    """Deep Generative Model of Radar"""

    def __init__(
        self,
        forecast_steps: int = 18,
        input_channels: int = 1,
        output_shape: int = 256,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        visualize: bool = False,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 768,
        context_channels: int = 384,
        **kwargs,
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels

        Args:
            forecast_steps: Number of steps to predict in the future
            input_channels: Number of input channels per image
            visualize: Whether to visualize output during training
            gen_lr: Learning rate for the generator
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial discriminator
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            num_samples: Number of samples of the latent space to sample for training/validation
            grid_lambda: Lambda for the grid regularization loss
            output_shape: Shape of the output predictions, generally should be same as the input shape
            latent_channels: Number of channels that the latent space should be reshaped to,
                input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs
            pretrained:
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        input_channels = self.config["input_channels"]
        forecast_steps = self.config["forecast_steps"]
        output_shape = self.config["output_shape"]
        gen_lr = self.config["gen_lr"]
        disc_lr = self.config["disc_lr"]
        conv_type = self.config["conv_type"]
        num_samples = self.config["num_samples"]
        grid_lambda = self.config["grid_lambda"]
        beta1 = self.config["beta1"]
        beta2 = self.config["beta2"]
        latent_channels = self.config["latent_channels"]
        context_channels = self.config["context_channels"]
        visualize = self.config["visualize"]
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss()
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.visualize = visualize
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=self.context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            shape=(8 * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )
        self.generator = Generator(self.conditioning_stack, self.latent_stack, self.sampler)
        self.discriminator = Discriminator(input_channels)
        self.save_hyperparameters()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch

        self.global_iteration += 1
        g_opt, d_opt = self.optimizers()
        ##########################
        # Optimize Discriminator #
        ##########################
        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = self(images)
            # Cat along time dimension [B, T, C, H, W]
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            # Cat long batch for the real+generated
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            concatenated_outputs = self.discriminator(concatenated_inputs)

            score_real, score_generated = torch.split(concatenated_outputs, 1, dim=1)
            discriminator_loss = loss_hinge_disc(score_generated, score_real)
            d_opt.zero_grad()
            self.manual_backward(discriminator_loss)
            d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        predictions = [self(images) for _ in range(6)]
        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)
        # Concat along time dimension
        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)
        # Cat long batch for the real+generated, for each example in the range
        # For each of the 6 examples
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = torch.split(concatenated_outputs, 1, dim=1)
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg
        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()

        self.log_dict(
            {
                "train/d_loss": discriminator_loss,
                "train/g_loss": generator_loss,
                "train/grid_loss": grid_cell_reg,
            },
            prog_bar=True,
        )

        # generate images
        generated_images = self(images)
        # log sampled images
        if self.visualize:
            self.visualize_step(
                images, future_images, generated_images, self.global_iteration, step="train"
            )

    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))

        return [opt_g, opt_d], []

    def visualize_step(
        self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor, batch_idx: int, step: str
    ) -> None:
        # the logger you used (in this case tensorboard)
        tensorboard = self.logger.experiment[0]
        # Timesteps per channel
        images = x[0].cpu().detach()
        future_images = y[0].cpu().detach()
        generated_images = y_hat[0].cpu().detach()
        for i, t in enumerate(images):  # Now would be (C, H, W)
            t = [torch.unsqueeze(img, dim=0) for img in t]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Input_Image_Stack_Frame_{i}", image_grid, global_step=batch_idx
            )
            t = [torch.unsqueeze(img, dim=0) for img in future_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Target_Image_Frame_{i}", image_grid, global_step=batch_idx
            )
            t = [torch.unsqueeze(img, dim=0) for img in generated_images[i]]
            image_grid = torchvision.utils.make_grid(t, nrow=self.input_channels)
            tensorboard.add_image(
                f"{step}/Generated_Image_Frame_{i}", image_grid, global_step=batch_idx
            )
