import torch
from nowcasting_gan import DGMR, Generator, Discriminator, TemporalDiscriminator, \
    SpatialDiscriminator, Sampler, LatentConditioningStack, ContextConditioningStack

def test_latent_conditioning_stack():
    model = LatentConditioningStack()
    x = torch.rand((2, 4, 1, 128, 128))
    out = model(x)
    assert out.size() == (
        1,
        768,
        8,
        8
        )
    assert not torch.isnan(out).any(), "Output included NaNs"

def test_context_conditioning_stack():
    model = ContextConditioningStack()
    x = torch.rand((2, 4, 1, 128, 128))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert len(out) == 4
    assert out[0].size() == (
        2,
        96,
        32,
        32
        )
    assert out[1].size() == (
        2,
        192,
        16,
        16
        )
    assert out[2].size() == (
        2,
        384,
        8,
        8
        )
    assert out[3].size() == (
        2,
        768,
        4,
        4
        )
    assert not all(torch.isnan(out[i]).any() for i in range(len(out))), "Output included NaNs"


def test_temporal_discriminator():
    model = TemporalDiscriminator(input_channels = 1)
    x = torch.rand((2, 8, 1, 256, 256))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 1)
    assert not torch.isnan(out).any()


def test_spatial_discriminator():
    model = SpatialDiscriminator(input_channels = 1)
    x = torch.rand((2, 18, 1, 128, 128))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 1)
    assert not torch.isnan(out).any()


def test_discriminator():
    model = Discriminator(input_channels = 1)
    x = torch.rand((2, 18, 1, 256, 256))
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 1)
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
