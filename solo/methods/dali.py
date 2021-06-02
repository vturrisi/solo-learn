import math
import os
from abc import ABC

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from solo.utils.dali_dataloader import (
    ContrastivePipeline,
    ImagenetTransform,
    MulticropContrastivePipeline,
    NormalPipeline,
)


class BaseWrapper(DALIGenericIterator):
    # this might be a shitty fix for now to handle when LastBatchPolicy.DROP is on
    def __len__(self):
        size = (
            self._size_no_pad // self._shards_num
            if self._last_batch_policy == LastBatchPolicy.DROP
            else self.size
        )
        if self._reader_name:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(size / self.batch_size)
            else:
                return size // self.batch_size
        else:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(size / (self._num_gpus * self.batch_size))
            else:
                return size // (self._num_gpus * self.batch_size)


class ContrastiveWrapper(BaseWrapper):
    def __init__(
        self,
        *args,
        model_batch_size=None,
        model_rank=None,
        model_device=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_batch_size = model_batch_size
        self.model_rank = model_rank
        self.model_device = model_device

    def __next__(self):
        batch = super().__next__()
        indexes = torch.arange(self.model_batch_size, device=self.model_device) + (
            self.model_rank * self.model_batch_size
        )
        *all_X, target = [batch[0][v] for v in self.output_map]
        target = target.squeeze(-1).long()
        return indexes, all_X, target


class Wrapper(BaseWrapper):
    def __next__(self):
        batch = super().__next__()
        x, target = batch[0]["x"], batch[0]["label"]
        target = target.squeeze(-1).long()
        return x, target


class ContrastiveABC(ABC):
    """
    Abstract contrastive class that returns a train_dataloader and val_dataloader using dali.
    """

    def train_dataloader(self):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        # get data arguments from model
        multicrop = self.extra_args["multicrop"]
        n_crops = self.extra_args["n_crops"]
        n_small_crops = self.extra_args["n_small_crops"]
        dali_device = self.extra_args["dali_device"]
        brightness = self.extra_args["brightness"]
        contrast = self.extra_args["contrast"]
        saturation = self.extra_args["saturation"]
        hue = self.extra_args["hue"]
        gaussian_prob = self.extra_args["gaussian_prob"]
        solarization_prob = self.extra_args["solarization_prob"]
        asymmetric_augmentations = self.extra_args["asymmetric_augmentations"]
        last_batch_fill = self.extra_args["last_batch_fill"]

        batch_size = self.extra_args["batch_size"]
        num_workers = self.extra_args["num_workers"]
        data_folder = self.extra_args["data_folder"]
        train_dir = self.extra_args["train_dir"]

        if multicrop:
            n_crops = [n_crops, n_small_crops]
            size_crops = [224, 96]
            min_scale_crops = [0.14, 0.05]
            max_scale_crops = [1.0, 0.14]

            transforms = []
            for size, min_scale, max_scale in zip(size_crops, min_scale_crops, max_scale_crops):
                transform = ImagenetTransform(
                    device=dali_device,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    gaussian_prob=gaussian_prob,
                    solarization_prob=solarization_prob,
                    size=size,
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
                transforms.append(transform)
            train_pipeline = MulticropContrastivePipeline(
                os.path.join(data_folder, train_dir),
                batch_size=batch_size,
                transforms=transforms,
                n_crops=n_crops,
                size_crops=size_crops,
                min_scale_crops=min_scale_crops,
                max_scale_crops=max_scale_crops,
                device=dali_device,
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                num_threads=num_workers,
            )
            output_map = [
                *[f"large{i}" for i in range(n_crops[0])],
                *[f"small{i}" for i in range(n_crops[1])],
                "label",
            ]

        else:
            min_scale_crop = self.extra_args["min_scale_crop"]

            if asymmetric_augmentations:
                transform = [
                    ImagenetTransform(
                        device=dali_device,
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                        gaussian_prob=1.0,
                        solarization_prob=0.0,
                        size=224,
                        min_scale=min_scale_crop,
                        max_scale=1.0,
                    ),
                    ImagenetTransform(
                        device=dali_device,
                        brightness=brightness,
                        contrast=contrast,
                        saturation=saturation,
                        hue=hue,
                        gaussian_prob=0.1,
                        solarization_prob=0.2,
                        size=224,
                        min_scale=min_scale_crop,
                        max_scale=1.0,
                    ),
                ]
            else:
                transform = ImagenetTransform(
                    device=dali_device,
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    gaussian_prob=gaussian_prob,
                    solarization_prob=solarization_prob,
                    size=224,
                    min_scale=min_scale_crop,
                    max_scale=1.0,
                )
            train_pipeline = ContrastivePipeline(
                os.path.join(data_folder, train_dir),
                batch_size=batch_size,
                transform=transform,
                device=dali_device,
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                num_threads=num_workers,
            )
            output_map = ["large1", "large2", "label"]

        policy = LastBatchPolicy.FILL if last_batch_fill else LastBatchPolicy.DROP
        train_loader = ContrastiveWrapper(
            train_pipeline,
            output_map=output_map,
            reader_name="Reader",
            last_batch_policy=policy,
            auto_reset=True,
            model_batch_size=batch_size,
            model_rank=device_id,
            model_device=self.device,
        )
        return train_loader


class ClassificationABC(ABC):
    """
    Abstract classification class that returns a train_dataloader and val_dataloader using dali.
    """

    def train_dataloader(self):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        batch_size = self.extra_args["batch_size"]
        num_workers = self.extra_args["num_workers"]
        dali_device = self.extra_args["dali_device"]
        data_folder = self.extra_args["data_folder"]
        train_dir = self.extra_args["train_dir"]

        train_pipeline = NormalPipeline(
            os.path.join(data_folder, train_dir),
            validation=False,
            batch_size=batch_size,
            device=dali_device,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            num_threads=num_workers,
        )
        train_loader = Wrapper(
            train_pipeline,
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )
        return train_loader

    def val_dataloader(self):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        batch_size = self.extra_args["batch_size"]
        num_workers = self.extra_args["num_workers"]
        dali_device = self.extra_args["dali_device"]
        data_folder = self.extra_args["data_folder"]
        val_dir = self.extra_args["val_dir"]

        val_pipeline = NormalPipeline(
            os.path.join(data_folder, val_dir),
            validation=True,
            batch_size=batch_size,
            device=dali_device,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            num_threads=num_workers,
        )

        val_loader = Wrapper(
            val_pipeline,
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        return val_loader
