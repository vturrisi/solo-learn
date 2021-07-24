import math
import os
from abc import ABC

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from solo.utils.dali_dataloader import (
    PretrainPipeline,
    ImagenetTransform,
    MulticropPretrainPipeline,
    NormalPipeline,
)


class BaseWrapper(DALIGenericIterator):
    """Temporary fix to handle LastBatchPolicy.DROP"""

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


class PretrainWrapper(BaseWrapper):
    def __init__(
        self,
        model_batch_size: int,
        model_rank: int,
        model_device: str,
        *args,
        **kwargs,
    ):
        """Adds indices to a batch fetched from the parent.

        Args:
            model_batch_size (int): batch size.
            model_rank (int): rank of the current process.
            model_device (str): id of the current device.
        """

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


class PretrainABC(ABC):
    """Abstract pretrain class that returns a train_dataloader using dali."""

    def train_dataloader(self) -> DALIGenericIterator:
        """Returns a train dataloader using dali. Supports multi-crop and asymmetric augmentations.

        Returns:
            DALIGenericIterator: a train dataloader in the form of a dali pipeline object wrapped
                with PretrainWrapper.
        """

        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        # get data arguments from model
        dali_device = self.extra_args["dali_device"]
        last_batch_fill = self.extra_args["last_batch_fill"]

        # data augmentations
        self.brightness = self.extra_args["brightness"]
        self.contrast = self.extra_args["contrast"]
        self.saturation = self.extra_args["saturation"]
        self.hue = self.extra_args["hue"]
        self.gaussian_prob = self.extra_args["gaussian_prob"]
        self.solarization_prob = self.extra_args["solarization_prob"]
        self.min_scale = self.extra_args["min_scale"]

        num_workers = self.extra_args["num_workers"]
        data_dir = self.extra_args["data_dir"]
        train_dir = self.extra_args["train_dir"]

        unique_augs = max(
            len(p)
            for p in [
                self.brightness,
                self.contrast,
                self.saturation,
                self.hue,
                self.gaussian_prob,
                self.solarization_prob,
                self.min_scale,
            ]
        )

        assert unique_augs == self.n_crops or unique_augs == 1

        # assert that either all unique augmentation pipelines have a unique
        # parameter or that a single parameter is replicated to all pipelines
        for p in [
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "gaussian_prob",
            "solarization_prob",
            "min_scale",
        ]:
            values = getattr(self, p)
            n = len(values)
            assert n == unique_augs or n == 1

            if n == 1:
                setattr(self, p, getattr(self, p) * unique_augs)

        if self.multicrop:
            n_crops = [self.n_crops, self.n_small_crops]
            size_crops = [224, 96]
            min_scales = [0.14, 0.05]
            max_scale_crops = [1.0, 0.14]

            transforms = []
            for size, min_scale, max_scale in zip(size_crops, min_scales, max_scale_crops):
                transform = ImagenetTransform(
                    device=dali_device,
                    **self.extra_args["transform_kwargs"],
                    size=size,
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
                transforms.append(transform)
            train_pipeline = MulticropPretrainPipeline(
                os.path.join(data_dir, train_dir),
                batch_size=self.batch_size,
                transforms=transforms,
                n_crops=n_crops,
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
            if unique_augs > 1:
                transform = [
                    ImagenetTransform(
                        device=dali_device,
                        **kwargs,
                        size=224,
                        max_scale=1.0,
                    )
                    for kwargs in self.extra_args["transform_kwargs"]
                ]

            else:
                transform = ImagenetTransform(
                    device=dali_device,
                    **self.extra_args["transform_kwargs"],
                    size=224,
                    max_scale=1.0,
                )
            train_pipeline = PretrainPipeline(
                os.path.join(data_dir, train_dir),
                batch_size=self.batch_size,
                transform=transform,
                device=dali_device,
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                num_threads=num_workers,
            )
            output_map = ["large1", "large2", "label"]

        policy = LastBatchPolicy.FILL if last_batch_fill else LastBatchPolicy.DROP
        train_loader = PretrainWrapper(
            model_batch_size=self.batch_size,
            model_rank=device_id,
            model_device=self.device,
            pipelines=train_pipeline,
            output_map=output_map,
            reader_name="Reader",
            last_batch_policy=policy,
            auto_reset=True,
        )
        return train_loader


class ClassificationABC(ABC):
    """Abstract classification class that returns a train_dataloader and val_dataloader using
    dali."""

    def train_dataloader(self) -> DALIGenericIterator:
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        num_workers = self.extra_args["num_workers"]
        dali_device = self.extra_args["dali_device"]
        data_dir = self.extra_args["data_dir"]
        train_dir = self.extra_args["train_dir"]

        train_pipeline = NormalPipeline(
            os.path.join(data_dir, train_dir),
            validation=False,
            batch_size=self.batch_size,
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

    def val_dataloader(self) -> DALIGenericIterator:
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        num_workers = self.extra_args["num_workers"]
        dali_device = self.extra_args["dali_device"]
        data_dir = self.extra_args["data_dir"]
        val_dir = self.extra_args["val_dir"]

        val_pipeline = NormalPipeline(
            os.path.join(data_dir, val_dir),
            validation=True,
            batch_size=self.batch_size,
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
