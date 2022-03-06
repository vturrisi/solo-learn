# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
from abc import ABC
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from solo.utils.dali_dataloader import (
    CustomNormalPipeline,
    CustomTransform,
    ImagenetTransform,
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

            return size // self.batch_size
        else:
            if self._last_batch_policy != LastBatchPolicy.DROP:
                return math.ceil(size / (self._num_gpus * self.batch_size))

            return size // (self._num_gpus * self.batch_size)


class PretrainWrapper(BaseWrapper):
    def __init__(
        self,
        model_batch_size: int,
        model_rank: int,
        model_device: str,
        conversion_map: List[int] = None,
        *args,
        **kwargs,
    ):
        """Adds indices to a batch fetched from the parent.

        Args:
            model_batch_size (int): batch size.
            model_rank (int): rank of the current process.
            model_device (str): id of the current device.
            conversion_map  (List[int], optional): list of integers that map each index
                to a class label. If nothing is passed, no label mapping needs to be done.
                Defaults to None.
        """

        super().__init__(*args, **kwargs)
        self.model_batch_size = model_batch_size
        self.model_rank = model_rank
        self.model_device = model_device
        self.conversion_map = conversion_map
        if self.conversion_map is not None:
            self.conversion_map = torch.tensor(
                self.conversion_map, dtype=torch.float32, device=self.model_device
            ).reshape(-1, 1)
            self.conversion_map = nn.Embedding.from_pretrained(self.conversion_map)

    def __next__(self):
        batch = super().__next__()[0]
        # PyTorch Lightning does double buffering
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1316,
        # and as DALI owns the tensors it returns the content of it is trashed so the copy needs,
        # to be made before returning.

        if self.conversion_map is not None:
            *all_X, indexes = [batch[v] for v in self.output_map]
            targets = self.conversion_map(indexes).flatten().long().detach().clone()
            indexes = indexes.flatten().long().detach().clone()
        else:
            *all_X, targets = [batch[v] for v in self.output_map]
            targets = targets.squeeze(-1).long().detach().clone()
            # creates dummy indexes
            indexes = (
                (
                    torch.arange(self.model_batch_size, device=self.model_device)
                    + (self.model_rank * self.model_batch_size)
                )
                .detach()
                .clone()
            )

        all_X = [x.detach().clone() for x in all_X]
        return [indexes, all_X, targets]


class Wrapper(BaseWrapper):
    def __next__(self):
        batch = super().__next__()
        x, target = batch[0]["x"], batch[0]["label"]
        target = target.squeeze(-1).long()
        # PyTorch Lightning does double buffering
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1316,
        # and as DALI owns the tensors it returns the content of it is trashed so the copy needs,
        # to be made before returning.
        x = x.detach().clone()
        target = target.detach().clone()
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
        num_crops_per_aug = self.extra_args["num_crops_per_aug"]

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

        if unique_augs > 1:
            transforms = [
                transform_pipeline(
                    device=dali_device,
                    **kwargs,
                )
                for kwargs in transform_kwargs
            ]
        else:
            transforms = [transform_pipeline(device=dali_device, **transform_kwargs)]

        train_pipeline = PretrainPipeline(
            data_dir / train_dir,
            batch_size=self.batch_size,
            transforms=transforms,
            num_crops_per_aug=num_crops_per_aug,
            device=dali_device,
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            num_threads=num_workers,
            no_labels=self.extra_args["no_labels"],
            encode_indexes_into_labels=self.encode_indexes_into_labels,
        )
        output_map = (
            [f"large{i}" for i in range(self.num_large_crops)]
            + [f"small{i}" for i in range(self.num_small_crops)]
            + ["label"]
        )

        policy = LastBatchPolicy.DROP
        conversion_map = train_pipeline.conversion_map if self.encode_indexes_into_labels else None
        train_loader = PretrainWrapper(
            model_batch_size=self.batch_size,
            model_rank=device_id,
            model_device=self.device,
            conversion_map=conversion_map,
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

        # handle custom data by creating the needed pipeline
        dataset = self.extra_args["dataset"]
        if dataset in ["imagenet100", "imagenet"]:
            pipeline_class = NormalPipeline
        elif dataset == "custom":
            pipeline_class = CustomNormalPipeline
        else:
            raise ValueError(dataset, "is not supported, used [imagenet, imagenet100 or custom]")

        train_pipeline = pipeline_class(
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

        # handle custom data by creating the needed pipeline
        dataset = self.extra_args["dataset"]
        if dataset in ["imagenet100", "imagenet"]:
            pipeline_class = NormalPipeline
        elif dataset == "custom":
            pipeline_class = CustomNormalPipeline
        else:
            raise ValueError(dataset, "is not supported, used [imagenet, imagenet100 or custom]")

        val_pipeline = pipeline_class(
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
