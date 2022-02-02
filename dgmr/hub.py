"""
Originally Taken from https://github.com/rwightman/

https://github.com/rwightman/pytorch-image-models/
blob/acd6c687fd1c0507128f0ce091829b233c8560b9/timm/models/hub.py
"""

import json
import logging
import os
from functools import partial

import torch


try:
    from huggingface_hub import cached_download, hf_hub_url

    cached_download = partial(cached_download, library_name="dgmr")
except ImportError:
    hf_hub_url = None
    cached_download = None

from huggingface_hub import CONFIG_NAME, PYTORCH_WEIGHTS_NAME, ModelHubMixin, hf_hub_download

MODEL_CARD_MARKDOWN = """---
license: mit
tags:
- nowcasting
- forecasting
- timeseries
- remote-sensing
- gan
---

# {model_name}

## Model description

[More information needed]

## Intended uses & limitations

[More information needed]

## How to use

[More information needed]

## Limitations and bias

[More information needed]

## Training data

[More information needed]

## Training procedure

[More information needed]

## Evaluation results

[More information needed]

"""

_logger = logging.getLogger(__name__)


class NowcastingModelHubMixin(ModelHubMixin):
    """
    HuggingFace ModelHubMixin containing specific adaptions for Nowcasting models
    """

    def __init__(self, *args, **kwargs):
        """
        Mixin for pl.LightningModule and Hugging Face

        Mix this class with your pl.LightningModule class to easily push / download
        the model via the Hugging Face Hub

        Example::

            >>> from dgmr.hub import NowcastingModelHubMixin

            >>> class MyModel(nn.Module, NowcastingModelHubMixin):
            ...    def __init__(self, **kwargs):
            ...        super().__init__()
            ...        self.layer = ...
            ...    def forward(self, ...)
            ...        return ...

            >>> model = MyModel()
            >>> model.push_to_hub("mymodel") # Pushing model-weights to hf-hub

            >>> # Downloading weights from hf-hub & model will be initialized from those weights
            >>> model = MyModel.from_pretrained("username/mymodel")
        """

    def _create_model_card(self, path):
        model_card = MODEL_CARD_MARKDOWN.format(model_name=type(self).__name__)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(model_card)

    def _save_config(self, module, save_directory):
        config = dict(module.hparams)
        path = os.path.join(save_directory, CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(config, f)

    def _save_pretrained(self, save_directory: str, save_config: bool = True):
        # Save model weights
        path = os.path.join(save_directory, PYTORCH_WEIGHTS_NAME)
        model_to_save = self.module if hasattr(self, "module") else self
        torch.save(model_to_save.state_dict(), path)
        # Save model config
        if save_config and model_to_save.hparams:
            self._save_config(model_to_save, save_directory)
        # Save model card
        self._create_model_card(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):
        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )
        model = cls(**model_kwargs["config"])

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model
