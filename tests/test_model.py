import torch
from nowcasting_gan import DGMR


def test_nowcasting_gan_creation():
    model = DGMR(
        forecast_steps=18,
        input_channels=1,
        output_shape=128,
        latent_channels=768,
        context_channels=384,
        num_samples=3,
    )
    x = torch.randn((2, 4, 1, 128, 128))
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
