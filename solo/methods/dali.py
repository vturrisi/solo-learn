import math
from abc import ABC
from pathlib import Path

import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from solo.utils.dali_dataloader import (
    CustomTransform,
    ImagenetTransform,
    MulticropPretrainPipeline,
    NormalPipeline,
    PretrainPipeline,
)


class BaseWrapper(DALIGenericIterator):
    """Temporary fix to handle LastBatchPolicy.DROP."""

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


def int_to_binary(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Converts a Tensor of integers to a Tensor in binary format.
    https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

    Args:
        x (torch.Tensor): tensor of interges to convert to binary format.
        bits (torch.Tensor): number of bits to use.

    Returns:
        torch.Tensor: x in binary format.
    """

    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def binary_to_int(b, bits):
    """Converts a Tensor in binary format to a Tensor of integers.
    https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits

    Args:
        x (torch.Tensor): tensor of binary data to convert to integer.
        bits (torch.Tensor): number of bits.

    Returns:
        torch.Tensor: x in integer format.
    """

    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1).detach()


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

        *all_X, targets = [batch[0][v] for v in self.output_map]
        targets = targets.squeeze(-1).long()
        # creates dummy indexes
        indexes = torch.arange(self.model_batch_size, device=self.model_device) + (
            self.model_rank * self.model_batch_size
        )

        return indexes, all_X, targets


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

        # data augmentations
        unique_augs = self.extra_args["unique_augs"]
        transform_kwargs = self.extra_args["transform_kwargs"]

        num_workers = self.extra_args["num_workers"]
        data_dir = Path(self.extra_args["data_dir"])
        train_dir = Path(self.extra_args["train_dir"])

        # hack to encode image indexes into the labels
        self.encode_indexes_into_labels = self.extra_args["encode_indexes_into_labels"]

        # handle custom data by creating the needed pipeline
        dataset = self.extra_args["dataset"]
        if dataset in ["imagenet100", "imagenet"]:
            transform_pipeline = ImagenetTransform
        elif dataset == "custom":
            transform_pipeline = CustomTransform
        else:
            raise ValueError(dataset, "is not supported, used [imagenet, imagenet100 or custom]")

        if self.multicrop:
            num_crops = [self.num_crops, self.num_small_crops]
            size_crops = [224, 96]
            min_scales = [0.14, 0.05]
            max_scale_crops = [1.0, 0.14]

            transforms = []
            for size, min_scale, max_scale in zip(size_crops, min_scales, max_scale_crops):
                transform = transform_pipeline(
                    device=dali_device,
                    **transform_kwargs,
                    size=size,
                    min_scale=min_scale,
                    max_scale=max_scale,
                )
                transforms.append(transform)
            train_pipeline = MulticropPretrainPipeline(
                data_dir / train_dir,
                batch_size=self.batch_size,
                transforms=transforms,
                num_crops=num_crops,
                device=dali_device,
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                num_threads=num_workers,
                no_labels=self.extra_args["no_labels"],
                encode_indexes_into_labels=self.encode_indexes_into_labels,
            )
            output_map = [
                *[f"large{i}" for i in range(num_crops[0])],
                *[f"small{i}" for i in range(num_crops[1])],
                "label",
            ]

        else:
            if unique_augs > 1:
                transform = [
                    transform_pipeline(
                        device=dali_device,
                        **kwargs,
                        max_scale=1.0,
                    )
                    for kwargs in transform_kwargs
                ]
            else:
                transform = transform_pipeline(
                    device=dali_device,
                    **transform_kwargs,
                    max_scale=1.0,
                )

            train_pipeline = PretrainPipeline(
                data_dir / train_dir,
                batch_size=self.batch_size,
                transform=transform,
                device=dali_device,
                device_id=device_id,
                shard_id=shard_id,
                num_shards=num_shards,
                num_threads=num_workers,
                no_labels=self.extra_args["no_labels"],
                encode_indexes_into_labels=self.encode_indexes_into_labels,
            )
            output_map = ["large1", "large2", "label"]

        policy = LastBatchPolicy.DROP
        train_loader = PretrainWrapper(
            model_batch_size=self.batch_size,
            model_rank=device_id,
            model_device=self.device,
            encode_indexes_into_labels=self.encode_indexes_into_labels,
            pipelines=train_pipeline,
            output_map=output_map,
            reader_name="Reader",
            last_batch_policy=policy,
            auto_reset=True,
        )

        self.dali_epoch_size = train_pipeline.epoch_size("Reader")

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
        data_dir = Path(self.extra_args["data_dir"])
        train_dir = Path(self.extra_args["train_dir"])

        train_pipeline = NormalPipeline(
            data_dir / train_dir,
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
        data_dir = Path(self.extra_args["data_dir"])
        val_dir = Path(self.extra_args["val_dir"])

        val_pipeline = NormalPipeline(
            data_dir / val_dir,
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
