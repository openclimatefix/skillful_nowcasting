import torch
import torch.nn.functional as F
from dgmr import (
    DGMR,
    Generator,
    Discriminator,
    TemporalDiscriminator,
    SpatialDiscriminator,
    Sampler,
    LatentConditioningStack,
    ContextConditioningStack,
)
from dgmr.layers import ConvGRU
from dgmr.layers.ConvGRU import ConvGRUCell
from dgmr.common import DBlock, GBlock
import einops
import pytest
from pytorch_lightning import Trainer


def test_dblock():
    model = DBlock(keep_same_output=True)
    x = torch.rand((2, 12, 128, 128))
    out = model(x)
    y = torch.rand((2, 12, 128, 128))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 12, 128, 128)
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_gblock():
    model = GBlock()
    x = torch.rand((2, 12, 128, 128))
    out = model(x)
    y = torch.rand((2, 12, 128, 128))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 12, 128, 128)
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_conv_gru_cell():
    model = ConvGRUCell(
        input_channels=768 + 384,
        output_channels=384,
        kernel_size=3,
    )
    x = torch.rand((2, 768, 32, 32))
    out, hidden = model(x, torch.rand((2, 384, 32, 32)))
    y = torch.rand((2, 384, 32, 32))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (2, 384, 32, 32)
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_conv_gru():
    model = ConvGRU(
        input_channels=768 + 384,
        output_channels=384,
        kernel_size=3,
    )
    init_states = [torch.rand((2, 384, 32, 32)) for _ in range(4)]
    # Expand latent dim to match batch size
    x = torch.rand((2, 768, 32, 32))
    hidden_states = [x] * 18
    out = model(hidden_states, init_states[3])
    y = torch.rand((18, 2, 384, 32, 32))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert out.size() == (18, 2, 384, 32, 32)
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_latent_conditioning_stack():
    model = LatentConditioningStack()
    x = torch.rand((2, 4, 1, 128, 128))
    out = model(x)
    assert out.size() == (1, 768, 8, 8)
    y = torch.rand((1, 768, 8, 8))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_context_conditioning_stack():
    model = ContextConditioningStack()
    x = torch.rand((2, 4, 1, 128, 128))
    out = model(x)
    y = torch.rand((2, 96, 32, 32))
    loss = F.mse_loss(y, out[0])
    loss.backward()
    assert len(out) == 4
    assert out[0].size() == (2, 96, 32, 32)
    assert out[1].size() == (2, 192, 16, 16)
    assert out[2].size() == (2, 384, 8, 8)
    assert out[3].size() == (2, 768, 4, 4)
    assert not all(torch.isnan(out[i]).any() for i in range(len(out))), "Output included NaNs"


def test_temporal_discriminator():
    model = TemporalDiscriminator(input_channels=1)
    x = torch.rand((2, 8, 1, 256, 256))
    out = model(x)
    assert out.shape == (2, 1, 1)
    y = torch.rand((2, 1, 1))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any()


def test_spatial_discriminator():
    model = SpatialDiscriminator(input_channels=1)
    x = torch.rand((2, 18, 1, 128, 128))
    out = model(x)
    assert out.shape == (2, 1, 1)
    y = torch.rand((2, 1, 1))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any()


def test_discriminator():
    model = Discriminator(input_channels=1)
    x = torch.rand((2, 18, 1, 256, 256))
    out = model(x)
    assert out.shape == (2, 2, 1)
    y = torch.rand((2, 2, 1))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any()


def test_sampler():
    input_channels = 1
    conv_type = "standard"
    context_channels = 384
    latent_channels = 768
    forecast_steps = 18
    output_shape = 256
    conditioning_stack = ContextConditioningStack(
        input_channels=input_channels,
        conv_type=conv_type,
        output_channels=context_channels,
    )
    latent_stack = LatentConditioningStack(
        shape=(8 * input_channels, output_shape // 32, output_shape // 32),
        output_channels=latent_channels,
    )
    sampler = Sampler(
        forecast_steps=forecast_steps,
        latent_channels=latent_channels,
        context_channels=context_channels,
    )
    latent_stack.eval()
    conditioning_stack.eval()
    sampler.eval()
    x = torch.rand((2, 4, 1, 256, 256))
    with torch.no_grad():
        latent_dim = latent_stack(x)
        assert not torch.isnan(latent_dim).any()
        init_states = conditioning_stack(x)
        assert not all(torch.isnan(init_states[i]).any() for i in range(len(init_states)))
        # Expand latent dim to match batch size
        latent_dim = einops.repeat(
            latent_dim, "b c h w -> (repeat b) c h w", repeat=init_states[0].shape[0]
        )
        assert not torch.isnan(latent_dim).any()
        hidden_states = [latent_dim] * forecast_steps
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = sampler.convGRU1(hidden_states, init_states[3])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        # Layer 3.
        hidden_states = sampler.convGRU2(hidden_states, init_states[2])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1_2(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g2(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g2(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))

        # Layer 2.
        hidden_states = sampler.convGRU3(hidden_states, init_states[1])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1_3(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g3(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g3(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))

        # Layer 1 (top-most).
        hidden_states = sampler.convGRU4(hidden_states, init_states[0])
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.gru_conv_1x1_4(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.g4(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.up_g4(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))

        # Output layer.
        hidden_states = [F.relu(sampler.bn(h)) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.conv_1x1(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))
        hidden_states = [sampler.depth2space(h) for h in hidden_states]
        assert not all(torch.isnan(hidden_states[i]).any() for i in range(len(hidden_states)))


def test_generator():
    input_channels = 1
    conv_type = "standard"
    context_channels = 384
    latent_channels = 768
    forecast_steps = 18
    output_shape = 256
    conditioning_stack = ContextConditioningStack(
        input_channels=input_channels,
        conv_type=conv_type,
        output_channels=context_channels,
    )
    latent_stack = LatentConditioningStack(
        shape=(8 * input_channels, output_shape // 32, output_shape // 32),
        output_channels=latent_channels,
    )
    sampler = Sampler(
        forecast_steps=forecast_steps,
        latent_channels=latent_channels,
        context_channels=context_channels,
    )
    model = Generator(
        conditioning_stack=conditioning_stack,
        latent_stack=latent_stack,
        sampler=sampler,
    )
    x = torch.rand((2, 4, 1, 256, 256))
    out = model(x)
    assert out.shape == (2, 18, 1, 256, 256)
    y = torch.rand((2, 18, 1, 256, 256))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any()


def test_nowcasting_gan_creation():
    model = DGMR(
        forecast_steps=18,
        input_channels=1,
        output_shape=128,
        latent_channels=768,
        context_channels=384,
        num_samples=3,
    )
    x = torch.rand((2, 4, 1, 128, 128))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.size() == (
        2,
        18,
        1,
        128,
        128,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_nowcasting_gan_backward():
    model = DGMR(
        forecast_steps=4,
        input_channels=1,
        output_shape=128,
        latent_channels=384,
        context_channels=192,
        num_samples=3,
    )
    x = torch.rand((2, 4, 1, 128, 128))
    out = model(x)
    assert out.size() == (
        2,
        4,
        1,
        128,
        128,
    )
    y = torch.rand((2, 4, 1, 128, 128))
    loss = F.mse_loss(y, out)
    loss.backward()
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_load_dgmr_from_hf():
    model = DGMR().from_pretrained("openclimatefix/dgmr")

@pytest.mark.skip("Takes too long")
def test_train_dgmr():
    forecast_steps = 8

    class DS(torch.utils.data.Dataset):
        def __init__(self, bs=2):
            self.ds = torch.rand((bs, forecast_steps + 4, 1, 256, 256))

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            return (self.ds[idx, 0:4, :, :], self.ds[idx, 4 : 4 + forecast_steps, :, :])

    train_loader = torch.utils.data.DataLoader(DS(), batch_size=1)
    val_loader = torch.utils.data.DataLoader(DS(), batch_size=1)

    trainer = Trainer(gpus=0, max_epochs=1)
    model = DGMR(forecast_steps=forecast_steps)

    trainer.fit(model, train_loader, val_loader)
