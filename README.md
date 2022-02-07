# Skillful Nowcasting with Deep Generative Model of Radar (DGMR)
Implementation of DeepMind's Skillful Nowcasting GAN Deep Generative Model of Radar (DGMR) (https://arxiv.org/abs/2104.00954) in PyTorch Lightning.

This implementation matches as much as possible the pseudocode released by DeepMind. Each of the components (Sampler, Context conditioning stack, Latent conditioning stack, Discriminator, and Generator) are normal PyTorch modules, as the training is a bit complicated, that is wrapped in PyTorch Lightning.

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
