"""Train the model on the dataset."""

from pathlib import Path

import numpy as np
import torch.utils.data.dataset
import wandb
from datasets import load_dataset
from numpy.random import default_rng
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

from dgmr import DGMR


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Return the wandb logger from the trainer."""
    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning \
            disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Class for the watch model."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        """Initialize the log frequency and log name."""
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Initialize the logger at the start of training."""
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True)


class UploadCheckpointsAsArtifact(Callback):
    """class for logging the checkpoint as artifacts."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        """Initialize the checkpoint directory."""
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        """Run when the user interupts the training by pressing a key on the keyboard."""
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Log information about the training of the module at the end of training."""
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log information about the validation of the model at the end of each validation epoch."""
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Log information about the training of the model at the end of each training epoch."""
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


class TFDataset(torch.utils.data.dataset.Dataset):
    """Class for representing a tensorflow dataset."""

    def __init__(self, split):
        """Load the dataset and the its split type."""
        super().__init__()
        self.reader = load_dataset(
            "openclimatefix/nimrod-uk-1km",
            "sample",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        self.iter_reader = self.reader

    def __len__(self):
        """Return the size of the TF dataset."""
        return 1000

    def __getitem__(self, item):
        """Return the next row of the loaded dataset."""
        try:
            row = next(self.iter_reader)
        except Exception:
            rng = default_rng()
            self.iter_reader = iter(
                self.reader.shuffle(seed=rng.integers(low=0, high=100000), buffer_size=1000)
            )
            row = next(self.iter_reader)
        input_frames, target_frames = extract_input_and_target_frames(row["radar_frames"])
        return np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1]), np.moveaxis(
            target_frames, [0, 1, 2, 3], [0, 2, 3, 1]
        )


class DGMRDataModule(LightningDataModule):
    """
    Example of LightningDataModule for NETCDF dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        num_workers: int = 1,
        pin_memory: bool = True,
        batch_size: int = 16,
    ):
        """fake_data: random data is created and used instead. This is useful for testing."""
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        self.dataloader_config = dict(
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=8,
            persistent_workers=True,
            # Disable automatic batching because dataset
            # returns complete batches.
            batch_size=None,
        )

    def train_dataloader(self):
        """Load the training dataset."""
        dataloader = DataLoader(
            TFDataset(split="train"), batch_size=self.batch_size, num_workers=self.num_workers
        )
        return dataloader

    def val_dataloader(self):
        """Load the validation dataset."""
        train_dataset = TFDataset(
            split="validation",
        )
        dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return dataloader


if __name__ == "__main__":
    wandb.init(project="dgmr")
    wandb_logger = WandbLogger(logger="dgmr")
    model_checkpoint = ModelCheckpoint(
        monitor="train/g_loss",
        dirpath="./",
        filename="best",
    )

    trainer = Trainer(
        max_epochs=1000,
        logger=wandb_logger,
        callbacks=[model_checkpoint],
        accelerator="auto",
        precision=32,
        # accelerator="tpu", devices=8
    )
    model = DGMR()
    datamodule = DGMRDataModule()
    trainer.fit(model, datamodule)
