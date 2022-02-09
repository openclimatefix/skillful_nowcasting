# Skillful Nowcasting with Deep Generative Model of Radar (DGMR)
Implementation of DeepMind's Skillful Nowcasting GAN Deep Generative Model of Radar (DGMR) (https://arxiv.org/abs/2104.00954) in PyTorch Lightning.

This implementation matches as much as possible the pseudocode released by DeepMind. Each of the components (Sampler, Context conditioning stack, Latent conditioning stack, Discriminator, and Generator) are normal PyTorch modules. As the model training is a bit complicated, the overall architecture is wrapped in PyTorch Lightning.

The default parameters match what is written in the paper.

## Installation

Clone the repository, then run
```shell
pip install -r requirements.txt
pip install -e .
````

Alternatively, you can also install through ```pip install dgmr```

## Training Data

The open-sourced UK training dataset is being added to [HuggingFace Datasets!](https://huggingface.co/datasets/openclimatefix/nimrod-uk-1km) This should enable training the original architecture on the original data for reproducing the results from the paper. Once the dataset is fully added, correctly pre-trained weights will be uploaded to the HF Hub too.

The dataset can be loaded with

```python
from datasets import load_dataset

dataset = load_dataset("openclimatefix/nimrod-uk-1km")
```

It is roughly 1Tb of space, so if you want to stream in the data instead of downloading it to disk, you can do

```python
from datasets import load_dataset

dataset = load_dataset("openclimatefix/nimrod-uk-1km", streaming=True)
```


## Pretrained Weights

Pretrained weights will be available soon through [HuggingFace Hub](https://huggingface.co/openclimatefix), currently random weights are available. The whole DGMR model or different components can be loaded as the following:

```python
from dgmr import DGMR, Sampler, Generator, Discriminator, LatentConditioningStack, ContextConditioningStack
model = DGMR().from_pretrained("openclimatefix/dgmr")
sampler = Sampler().from_pretrained("openclimatefix/dgmr-sampler")
generator = Generator().from_pretrained("openclimatefix/dgmr-generator")
discriminator = Discriminator().from_pretrained("openclimagefix/dgmr-discriminator")
latent_stack = LatentConditioningStack().from_pretrained("openclimatefix/dgmr-latent-conditioning-stack")
context_stack = ContextConditioningStack().from_pretrained("openclimatefix/dgmr-context-conditioning-stack")
```

## Example Usage

```python
from dgmr import DGMR
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
y = torch.rand((2, 4, 1, 128, 128))
loss = F.mse_loss(y, out)
loss.backward()
```
