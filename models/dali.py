import math
import os
import sys

import torch

try:
    from linear import LinearModel
    from simclr import SimCLR
    from barlow_twins import BarlowTwins
    from simsiam import SimSiam
except:
    from .linear import LinearModel
    from .simclr import SimCLR
    from .barlow_twins import BarlowTwins
    from .simsiam import SimSiam

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from abc import ABC

from utils.dali_dataloader import ContrastivePipeline, NormalPipeline


class ContrastiveWrapper(DALIGenericIterator):
    def __init__(
        self, *args, model_batch_size=None, model_rank=None, model_device=None, **kwargs,
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


class Wrapper(DALIGenericIterator):
    def __init__(self, *kargs, **kvargs):
        super().__init__(*kargs, **kvargs)

    def __next__(self):
        batch = super().__next__()
        x, target = batch[0]["x"], batch[0]["label"]
        target = target.squeeze(-1).long()
        return x, target


class ContrastiveABC(ABC):
    """
    Abstract contrastive class that returns a train_dataloader and val_dataloader using dali.
    """

    def setup(self, stage=None):
        device_id = self.local_rank
        num_shards = self.trainer.world_size

        if self.args.multicrop:
            nmb_crops = [self.args.n_crops, self.args.n_small_crops]
            size_crops = [224, 96]
            min_scale_crops = [0.14, 0.05]
            max_scale_crops = [1.0, 0.14]
            self.output_map = [
                *[f"large{i}" for i in range(nmb_crops[0])],
                *[f"small{i}" for i in range(nmb_crops[1])],
                "label",
            ]
        else:
            nmb_crops = [self.args.n_crops]
            size_crops = [224]
            min_scale_crops = [0.2]
            max_scale_crops = [1.0]
            self.output_map = [
                *[f"large{i}" for i in range(nmb_crops[0])],
                "label",
            ]

        train_pipeline = ContrastivePipeline(
            os.path.join(self.args.data_folder, self.args.train_dir),
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            batch_size=self.args.batch_size,
            brightness=self.args.brightness,
            contrast=self.args.contrast,
            saturation=self.args.saturation,
            hue=self.args.hue,
            device="gpu",
            device_id=device_id,
            num_shards=num_shards,
            num_threads=self.args.num_workers,
        )
        policy = LastBatchPolicy.FILL if self.args.last_batch_fill else LastBatchPolicy.DROP
        self.train_loader = ContrastiveWrapper(
            train_pipeline,
            output_map=self.output_map,
            reader_name="Reader",
            last_batch_policy=policy,
            auto_reset=True,
            model_batch_size=self.args.batch_size,
            model_rank=device_id,
            model_device=self.device,
        )

        val_pipeline = NormalPipeline(
            os.path.join(self.args.data_folder, self.args.val_dir),
            validation=True,
            batch_size=self.args.batch_size,
            device="gpu",
            device_id=device_id,
            num_shards=num_shards,
            num_threads=self.args.num_workers,
        )
        self.val_loader = Wrapper(
            val_pipeline,
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class ClassificationABC(ABC):
    """
    Abstract classification class that returns a train_dataloader and val_dataloader using dali.
    """

    def setup(self, stage=None):
        device_id = self.local_rank
        num_shards = self.trainer.world_size

        train_pipeline = NormalPipeline(
            os.path.join(self.args.data_folder, self.args.train_dir),
            validation=False,
            batch_size=self.args.batch_size,
            device="gpu",
            device_id=device_id,
            num_shards=num_shards,
            num_threads=self.args.num_workers,
        )
        self.train_loader = Wrapper(
            train_pipeline,
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )

        val_pipeline = NormalPipeline(
            os.path.join(self.args.data_folder, self.args.val_dir),
            validation=True,
            batch_size=self.args.batch_size,
            device="gpu",
            device_id=device_id,
            num_shards=num_shards,
            num_threads=self.args.num_workers,
        )

        self.val_loader = Wrapper(
            val_pipeline,
            output_map=["x", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class DaliSimCLR(SimCLR, ContrastiveABC):
    pass


class DaliLinearModel(LinearModel, ClassificationABC):
    pass


class DaliBarlowTwins(BarlowTwins, ContrastiveABC):
    pass


class DaliSimSiam(SimSiam, ContrastiveABC):
    pass
