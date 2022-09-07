# Skillful Nowcasting with Deep Generative Model of Radar (DGMR)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
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

The authors also used [MRMS](https://www.nssl.noaa.gov/projects/mrms/) US precipitation radar data as another comparison. While that dataset was not released, the MRMS data is publicly available, and we have made that data available on HuggingFace Datasets as well [here](https://huggingface.co/datasets/openclimatefix/mrms). This dataset is the raw 3500x7000 contiguous US MRMS data for 2016 through May 2022, is a few hundred GBs in size, with sporadic updates to more recent data planned. This dataset is in Zarr format, and can be streamed without caching locally through 

```python
from datasets import load_dataset

dataset = load_dataset("openclimatefix/mrms", "default_sequence", streaming=True)
```

This steams the data with 24 timesteps per example, just like the UK DGMR dataset. To get individual MRMS frames, instead of a sequence, this can be achieved through 

```python
from datasets import load_dataset

dataset = load_dataset("openclimatefix/mrms", "default", streaming=True)
```

## Pretrained Weights

Pretrained weights are be available through [HuggingFace Hub](https://huggingface.co/openclimatefix), currently weights trained on the sample dataset. The whole DGMR model or different components can be loaded as the following:

```python
from dgmr import DGMR, Sampler, Generator, Discriminator, LatentConditioningStack, ContextConditioningStack
model = DGMR.from_pretrained("openclimatefix/dgmr")
sampler = Sampler.from_pretrained("openclimatefix/dgmr-sampler")
discriminator = Discriminator.from_pretrained("openclimatefix/dgmr-discriminator")
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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jacob Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/skillful_nowcasting/commits?author=jacobbieker" title="Code">💻</a></td>
    <td align="center"><a href="http://johmathe.name/"><img src="https://avatars.githubusercontent.com/u/467643?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Johan Mathe</b></sub></a><br /><a href="https://github.com/openclimatefix/skillful_nowcasting/commits?author=johmathe" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!