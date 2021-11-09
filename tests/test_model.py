import torch
from nowcasting_gan import (
    DGMR,
    Generator,
    Discriminator,
    TemporalDiscriminator,
    SpatialDiscriminator,
    Sampler,
    LatentConditioningStack,
    ContextConditioningStack,
)
from nowcasting_gan.layers import ConvGRU


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
    model.eval()
    with torch.no_grad():
        out = model(hidden_states, init_states[3])
    assert out.size() == (18, 2, 384, 32, 32)
    assert not torch.isnan(out).any(), "Output included NaNs"

def test_latent_conditioning_stack():
    model = LatentConditioningStack()
    x = torch.rand((2, 4, 1, 128, 128))
    out = model(x)
    assert out.size() == (1, 768, 8, 8)
    assert not torch.isnan(out).any(), "Output included NaNs"


def test_context_conditioning_stack():
    model = ContextConditioningStack()
    x = torch.rand((2, 4, 1, 128, 128))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert len(out) == 4
    assert out[0].size() == (2, 96, 32, 32)
    assert out[1].size() == (2, 192, 16, 16)
    assert out[2].size() == (2, 384, 8, 8)
    assert out[3].size() == (2, 768, 4, 4)
    assert not all(torch.isnan(out[i]).any() for i in range(len(out))), "Output included NaNs"


def test_temporal_discriminator():
    model = TemporalDiscriminator(input_channels=1)
    x = torch.rand((2, 8, 1, 256, 256))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 1)
    assert not torch.isnan(out).any()


def test_spatial_discriminator():
    model = SpatialDiscriminator(input_channels=1)
    x = torch.rand((2, 18, 1, 128, 128))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 1)
    assert not torch.isnan(out).any()


def test_discriminator():
    model = Discriminator(input_channels=1)
    x = torch.rand((2, 18, 1, 256, 256))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 1)
    assert not torch.isnan(out).any()

def test_generator():
    input_channels = 1
    conv_type = 'standard'
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
    model = Generator(conditioning_stack = conditioning_stack,
                      latent_stack = latent_stack,
                      sampler = sampler)
    x = torch.rand((2, 4, 1, 256, 256))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 18, 1, 256, 256)
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
