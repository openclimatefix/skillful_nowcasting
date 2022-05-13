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

The open-sourced UK training dataset has been mirrored to [HuggingFace Datasets!](https://huggingface.co/datasets/openclimatefix/nimrod-uk-1km) This should enable training the original architecture on the original data for reproducing the results from the paper. The full dataset is roughly 1TB in size, and unfortunately, streaming the data from HF Datasets doesn't seem to work, so it has to be cached locally. We have added the sample dataset as well though, which can be directly streamed from GCP without costs.

The dataset can be loaded with

```python
from datasets import load_dataset

dataset = load_dataset("openclimatefix/nimrod-uk-1km")
```

For now, only the sample dataset support streaming in, as its data files are hosted on GCP, not HF, so it can be used with:

```python
from datasets import load_dataset

dataset = load_dataset("openclimatefix/nimrod-uk-1km", "sample", streaming=True)
```


## Pretrained Weights

Pretrained weights will be available soon through [HuggingFace Hub](https://huggingface.co/openclimatefix), currently random weights are available. The whole DGMR model or different components can be loaded as the following:

```python
from dgmr import DGMR, Sampler, Generator, Discriminator, LatentConditioningStack, ContextConditioningStack
model = DGMR.from_pretrained("openclimatefix/dgmr")
sampler = Sampler.from_pretrained("openclimatefix/dgmr-sampler")
discriminator = Discriminator.from_pretrained("openclimagefix/dgmr-discriminator")
latent_stack = LatentConditioningStack.from_pretrained("openclimatefix/dgmr-latent-conditioning-stack")
context_stack = ContextConditioningStack.from_pretrained("openclimatefix/dgmr-context-conditioning-stack")
generator = Generator(conditioning_stack=context_stack, latent_stack=latent_stack, sampler=sampler)
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

## Citation
```
@article{ravuris2021skillful,
  author={Suman Ravuri and Karel Lenc and Matthew Willson and Dmitry Kangin and Remi Lam and Piotr Mirowski and Megan Fitzsimons and Maria Athanassiadou and Sheleem Kashem and Sam Madge and Rachel Prudden Amol Mandhane and Aidan Clark and Andrew Brock and Karen Simonyan and Raia Hadsell and Niall Robinson Ellen Clancy and Alberto Arribas† and Shakir Mohamed},
  title={Skillful Precipitation Nowcasting using Deep Generative Models of Radar},
  journal={Nature},
  volume={597},
  pages={672--677},
  year={2021}
}
```
