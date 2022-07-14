from dgmr import DGMR
from torch.utils.data import DataLoader
from pytorch_lightning import (
    LightningDataModule,
)
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
wandb.init(project="dgmr")
from pathlib import Path
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
from torch.utils.data import DataLoader, IterableDataset

precision = 16
DATASET_ROOT_DIR = "/mnt/leonardo/storage_ssd_8tb/data/ocf/dgmr/nimrod-uk-1km/20200718/"

"""
This loading code below is copyright DeepMind Technologies, licensed under Apache 2.0
"""
_FEATURES = {name: tf.io.FixedLenFeature([], dtype)
             for name, dtype in [
                 ("radar", tf.string), ("sample_prob", tf.float32),
                 ("osgb_extent_top", tf.int64), ("osgb_extent_left", tf.int64),
                 ("osgb_extent_bottom", tf.int64), ("osgb_extent_right", tf.int64),
                 ("end_time_timestamp", tf.int64),
             ]}

_SHAPE_BY_SPLIT_VARIANT = {
    ("train", "random_crops_256"): (24, 256, 256, 1),
    ("valid", "subsampled_tiles_256_20min_stride"): (24, 256, 256, 1),
    ("test", "full_frame_20min_stride"): (24, 1536, 1280, 1),
    ("test", "subsampled_overlapping_padded_tiles_512_20min_stride"): (24, 512, 512, 1),
}

_MM_PER_HOUR_INCREMENT = 1 / 32.
_MAX_MM_PER_HOUR = 128.
_INT16_MASK_VALUE = -1


def parse_and_preprocess_row(row, split, variant):
    """This loading code below is copyright DeepMind Technologies, licensed under Apache 2.0"""
    result = tf.io.parse_example(row, _FEATURES)
    shape = _SHAPE_BY_SPLIT_VARIANT[(split, variant)]
    radar_bytes = result.pop("radar")
    radar_int16 = tf.reshape(tf.io.decode_raw(radar_bytes, tf.int16), shape)
    mask = tf.not_equal(radar_int16, _INT16_MASK_VALUE)
    radar = tf.cast(radar_int16, tf.float32) * _MM_PER_HOUR_INCREMENT
    radar = tf.clip_by_value(
        radar, _INT16_MASK_VALUE * _MM_PER_HOUR_INCREMENT, _MAX_MM_PER_HOUR)
    result["radar_frames"] = radar
    result["radar_mask"] = mask
    return result


def reader(split, variant, shuffle_files: bool = False):
    """Reader for open-source nowcasting datasets.

    Copyright DeepMind Technologies, licensed under Apache 2.0
    Args:
      split: Which yearly split of the dataset to use:
        "train": Data from 2016 - 2018, excluding the first day of each month.
        "valid": Data from 2016 - 2018, only the first day of the month.
        "test": Data from 2019.
      variant: Which variant to use. The available variants depend on the split:
        "random_crops_256": Available for the training split. 24x256x256 pixel
          crops, sampled with a bias towards crops containing rainfall. Crops at
          all spatial and temporal offsets were able to be sampled, some crops may
          overlap.
        "subsampled_tiles_256_20min_stride": Available for the validation set.
          Non-spatially-overlapping 24x256x256 pixel crops, subsampled from a
          regular spatial grid with stride 256x256 pixels, and a temporal stride
          of 20mins (4 timesteps at 5 minute resolution). Sampling favours crops
          containing rainfall.
        "subsampled_overlapping_padded_tiles_512_20min_stride": Available for the
          test set. Overlapping 24x512x512 pixel crops, subsampled from a
          regular spatial grid with stride 64x64 pixels, and a temporal stride
          of 20mins (4 timesteps at 5 minute resolution). Subsampling favours
          crops containing rainfall.
          These crops include extra spatial context for a fairer evaluation of
          the PySTEPS baseline, which benefits from this extra context. Our other
          models only use the central 256x256 pixels of these crops.
        "full_frame_20min_stride": Available for the test set. Includes full
          frames at 24x1536x1280 pixels, every 20 minutes with no additional
          subsampling.
      shuffle_files: Whether to shuffle the shard files of the dataset
        non-deterministically before interleaving them. Recommended for the
        training set to improve mixing and read performance (since
        non-deterministic parallel interleave is then enabled).
    Returns:
      A tf.data.Dataset whose rows are dicts with the following keys:
      "radar_frames": Shape TxHxWx1, float32. Radar-based estimates of
        ground-level precipitation, in units of mm/hr. Pixels which are masked
        will take on a value of -1/32 and should be excluded from use as
        evaluation targets. The coordinate reference system used is OSGB36, with
        a spatial resolution of 1000 OSGB36 coordinate units (approximately equal
        to 1km). The temporal resolution is 5 minutes.
      "radar_mask": Shape TxHxWx1, bool. A binary mask which is False
        for pixels that are unobserved / unable to be inferred from radar
        measurements (e.g. due to being too far from a radar site). This mask
        is usually static over time, but occasionally a whole radar site will
        drop in or out resulting in large changes to the mask, and more localised
        changes can happen too.
      "sample_prob": Scalar float. The probability with which the row was
        sampled from the overall pool available for sampling, as described above
        under 'variants'. We use importance weights proportional to 1/sample_prob
        when computing metrics on the validation and test set, to reduce bias due
        to the subsampling.
      "end_time_timestamp": Scalar int64. A timestamp for the final frame in
        the example, in seconds since the UNIX epoch (1970-01-01 00:00:00 UTC).
      "osgb_extent_left", "osgb_extent_right", "osgb_extent_top",
      "osgb_extent_bottom":
        Scalar int64s. Spatial extent for the crop in the OSGB36 coordinate
        reference system.
    """
    shards_glob = os.path.join(DATASET_ROOT_DIR, split, variant, "*.tfrecord.gz")
    shard_paths = tf.io.gfile.glob(shards_glob)
    shards_dataset = tf.data.Dataset.from_tensor_slices(shard_paths)
    return (
        shards_dataset
        .interleave(lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=not shuffle_files)
        .map(lambda row: parse_and_preprocess_row(row, split, variant),
             num_parallel_calls=tf.data.AUTOTUNE)
        # Do your own subsequent repeat, shuffle, batch, prefetch etc as required.
    )

NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES-NUM_INPUT_FRAMES : -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES : ]
    return input_frames, target_frames

def process_data(example):
    input_frames, target_frames = extract_input_and_target_frames(example["radar_frames"])
    return {"input": np.moveaxis(input_frames, [0, 1, 2, 3], [0, 2, 3, 1]), "target": np.moveaxis(target_frames, [0, 1, 2, 3], [0, 2, 3, 1]),
            "mask": np.moveaxis(example["radar_mask"][-NUM_TARGET_FRAMES : ], [0, 1, 2, 3], [0, 2, 3, 1]),}



class TFDataset(IterableDataset):
    def __init__(self, split, variant):
        self.split = split
        self.variant = variant
        self.reader = reader(split=split, variant=variant)

    def __iter__(self):
        for key, row in enumerate(self.reader):
            data = {k: v.numpy() for k, v in row.items()}
            data = process_data(data)
            yield data["input"], data["target"]



def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
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
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


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
    ):
        """
        fake_data: random data is created and used instead. This is useful for testing
        """
        super().__init__()

        self.num_workers = num_workers
        self.pin_memory = pin_memory

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
        dataloader = DataLoader(TFDataset(split="train", variant="random_crops_256"), batch_size=4 if precision == 16 else 1, num_workers=16)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(TFDataset(split="valid", variant="subsampled_tiles_256_20min_stride"), batch_size=8 if precision == 16 else 1, num_workers=16)
        return dataloader


wandb_logger = WandbLogger(logger="dgmr")
model_checkpoint = ModelCheckpoint(
    monitor="val/g_loss",
    dirpath="./",
    filename="best",
)

trainer = Trainer(
    max_epochs=1000,
    logger=wandb_logger,
    callbacks=[model_checkpoint],
    gpus=1,
    precision=precision,
    #accelerator="tpu", devices=8
)
model = DGMR(grad_accumulate_steps=4 if precision == 16 else 16)
datamodule = DGMRDataModule()
trainer.fit(model, datamodule)
