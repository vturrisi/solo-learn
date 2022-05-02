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

import argparse

from pytorch_lightning import Trainer
from solo.methods import BarlowTwins
from solo.methods.linear import LinearModel
from solo.utils.dali_dataloader import (
    ClassificationDALIDataModule,
    ImagenetTransform,
    NormalPipelineBuilder,
    PretrainDALIDataModule,
    PretrainPipelineBuilder,
)
from torch import nn
from torchvision.models.resnet import resnet18

from ..methods.utils import DATA_KWARGS, gen_base_kwargs
from .utils import DummyDataset


def test_dali_dataloader():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("dummy_train", "dummy_val", 10, 4):
        size_crops = [224, 96]
        min_scales = [0.14, 0.05]
        max_scale_crops = [1.0, 0.14]

        transforms = []
        for size, min_scale, max_scale in zip(size_crops, min_scales, max_scale_crops):
            transform = ImagenetTransform(
                device="cpu",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
                gaussian_prob=0.5,
                solarization_prob=0.1,
                crop_size=size,
                min_scale=min_scale,
                max_scale=max_scale,
            )
            transforms.append(transform)

        # multicrop pipeline
        train_pipeline_builder = PretrainPipelineBuilder(
            "dummy_train",
            batch_size=4,
            transforms=transforms,
            num_crops_per_aug=[2, 4],
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        train_pipeline = train_pipeline_builder.pipeline(
            batch_size=train_pipeline_builder.batch_size,
            num_threads=train_pipeline_builder.num_threads,
            device_id=train_pipeline_builder.device_id,
            seed=train_pipeline_builder.seed,
        )
        train_pipeline.build()

        # normal pipeline
        train_pipeline_builder = NormalPipelineBuilder(
            "dummy_train",
            validation=False,
            batch_size=4,
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        train_pipeline = train_pipeline_builder.pipeline(
            batch_size=train_pipeline_builder.batch_size,
            num_threads=train_pipeline_builder.num_threads,
            device_id=train_pipeline_builder.device_id,
            seed=train_pipeline_builder.seed,
        )
        train_pipeline.build()

        val_pipeline_builder = NormalPipelineBuilder(
            "dummy_val",
            validation=True,
            batch_size=4,
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        val_pipeline = val_pipeline_builder.pipeline(
            batch_size=val_pipeline_builder.batch_size,
            num_threads=val_pipeline_builder.num_threads,
            device_id=val_pipeline_builder.device_id,
            seed=val_pipeline_builder.seed,
        )
        val_pipeline.build()


def test_dali_pretrain():
    for encode_indexes_into_labels in [True, False]:
        # creates a dummy dataset that autodeletes after usage
        with DummyDataset("dummy_train", "dummy_val", 128, 4):
            method_kwargs = {
                "proj_hidden_dim": 2048,
                "proj_output_dim": 2048,
                "lamb": 5e-3,
                "scale_loss": 0.025,
                "encode_indexes_into_labels": encode_indexes_into_labels,
                "disable_knn_eval": True,
            }
            BASE_KWARGS = gen_base_kwargs(cifar=False)
            DATA_KWARGS_WRAPPED = {k: [v] for k, v in DATA_KWARGS.items()}
            kwargs = {**BASE_KWARGS, **DATA_KWARGS_WRAPPED, **method_kwargs}

            kwargs["dali_device"] = "cpu"
            kwargs["train_dir"] = "dummy_train"
            kwargs["data_dir"] = "."
            kwargs["dataset"] = "custom"

            kwargs["transform_kwargs"] = dict(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
                gaussian_prob=0.5,
                solarization_prob=0.2,
                min_scale=0.08,
                max_scale=1.0,
            )
            kwargs["unique_augs"] = 1

            model = BarlowTwins(**kwargs)

            args = argparse.Namespace(**kwargs)
            trainer = Trainer.from_argparse_args(
                args,
                checkpoint_callback=False,
                limit_train_batches=2,
                limit_val_batches=2,
            )
            dali_datamodule = PretrainDALIDataModule(
                dataset=args.dataset,
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                unique_augs=args.unique_augs,
                transform_kwargs=args.transform_kwargs,
                num_crops_per_aug=args.num_crops_per_aug,
                num_large_crops=args.num_large_crops,
                num_small_crops=args.num_small_crops,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                no_labels=args.no_labels,
                data_fraction=args.data_fraction,
                dali_device=args.dali_device,
                encode_indexes_into_labels=args.encode_indexes_into_labels,
            )
            trainer.fit(model, datamodule=dali_datamodule)


def test_dali_linear():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("./dummy_train", "./dummy_val", 128, 4):
        BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True)
        kwargs = {**BASE_KWARGS, **DATA_KWARGS}
        backbone = resnet18()
        backbone.fc = nn.Identity()

        kwargs["dali_device"] = "cpu"
        kwargs["data_dir"] = "."
        kwargs["train_dir"] = "dummy_train"
        kwargs["val_dir"] = "dummy_val"
        kwargs["dataset"] = "custom"

        del kwargs["backbone"]

        model = LinearModel(backbone, **kwargs)

        args = argparse.Namespace(**kwargs)
        dali_datamodule = ClassificationDALIDataModule(
            dataset=args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            data_fraction=args.data_fraction,
            dali_device=args.dali_device,
        )

        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=False,
            limit_train_batches=2,
            limit_val_batches=2,
        )
        trainer.fit(model, datamodule=dali_datamodule)
