import torch
from nowcasting_gan import NowcastingGAN


def test_nowcasting_gan_creation():
    model = NowcastingGAN(
        forecast_steps=24,
        input_channels=1,
        output_shape=128,
        latent_channels=768,
        context_channels=768,
        num_samples=3,
    )
    x = torch.randn((2, 4, 1, 128, 128))
    model.eval()
    with torch.no_grad():
        out = model(x)
    # MetNet creates predictions for the center 1/4th
    assert out.size() == (
        2,
        24,
        1,
        128,
        128,
    )
    assert not torch.isnan(out).any(), "Output included NaNs"
