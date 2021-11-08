import tensorflow as tf
import tensorflow_hub
import torch
from nowcasting_gan import NowcastingGAN
import os
import fsspec

module = tensorflow_hub.load("/home/jacob/256x256/")
print(module)
print(module.signatures)
sig_model = module.signatures["default"]
