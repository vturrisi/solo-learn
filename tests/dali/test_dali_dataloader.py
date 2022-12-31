# Copyright 2023 solo-learn development team.

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

from omegaconf import OmegaConf
from solo.data.dali_dataloader import (
    ClassificationDALIDataModule,
    NormalPipelineBuilder,
    PretrainDALIDataModule,
    PretrainPipelineBuilder,
    build_transform_pipeline_dali,
)
from solo.data.pretrain_dataloader import FullTransformPipeline, NCropAugmentation
from solo.methods import BarlowTwins
from solo.methods.linear import LinearModel
from torch import nn
from torchvision.models.resnet import resnet18

from ..methods.utils import gen_base_cfg, gen_trainer
from .utils import DummyDataset


def test_dali_dataloader():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("dummy_train", "dummy_val", 10, 4):
        size_crops = [224, 96]
        min_scales = [0.14, 0.05]
        max_scale_crops = [1.0, 0.14]

        transforms = []
        for size, min_scale, max_scale in zip(size_crops, min_scales, max_scale_crops):
            cfg = OmegaConf.create(
                {
                    "crop_size": size,
                    "rrc": {
                        "enabled": True,
                        "crop_min_scale": min_scale,
                        "crop_max_scale": max_scale,
                    },
                    "color_jitter": {
                        "prob": 0.8,
                        "brightness": 0.4,
                        "contrast": 0.4,
                        "saturation": 0.4,
                        "hue": 0.2,
                    },
                    "grayscale": {"prob": 0.5},
                    "gaussian_blur": {"prob": 0.5},
                    "solarization": {"prob": 0.1},
                    "equalization": {"prob": 0.0},
                    "horizontal_flip": {"prob": 0.5},
                    "num_crops": 2,
                }
            )
            transforms.append(
                NCropAugmentation(
                    build_transform_pipeline_dali("imagenet100", cfg, dali_device="cpu"),
                    cfg.num_crops,
                )
            )

        transform = FullTransformPipeline(transforms)

        # multicrop pipeline
        train_pipeline_builder = PretrainPipelineBuilder(
            "dummy_train",
            batch_size=4,
            transforms=transform,
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
            }
            cfg = gen_base_cfg("barlow_twins", batch_size=2, num_classes=100)
            cfg.method_kwargs = method_kwargs
            cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)
            cfg.dali.device = "cpu"
            cfg.data.train_path = "./dummy_train"
            cfg.data.dataset = "custom"
            model = BarlowTwins(cfg)

            trainer = gen_trainer(cfg)

            aug_cfg = OmegaConf.create(
                {
                    "crop_size": 224,
                    "rrc": {
                        "enabled": True,
                        "crop_min_scale": 0.08,
                        "crop_max_scale": 1.0,
                    },
                    "color_jitter": {
                        "prob": 0.8,
                        "brightness": 0.4,
                        "contrast": 0.4,
                        "saturation": 0.4,
                        "hue": 0.2,
                    },
                    "grayscale": {"prob": 0.5},
                    "gaussian_blur": {"prob": 0.5},
                    "solarization": {"prob": 0.1},
                    "equalization": {"prob": 0.0},
                    "horizontal_flip": {"prob": 0.5},
                    "num_crops": 2,
                }
            )
            pipelines = [
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            ]
            transform = FullTransformPipeline(pipelines)
            dali_datamodule = PretrainDALIDataModule(
                dataset=cfg.data.dataset,
                train_data_path=cfg.data.train_path,
                transforms=transform,
                num_large_crops=cfg.data.num_large_crops,
                num_small_crops=cfg.data.num_small_crops,
                num_workers=cfg.data.num_workers,
                batch_size=cfg.optimizer.batch_size,
                no_labels=False,
                data_fraction=-1,
                dali_device=cfg.dali.device,
                encode_indexes_into_labels=encode_indexes_into_labels,
            )
            trainer.fit(model, datamodule=dali_datamodule)


def test_dali_linear():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("./dummy_train", "./dummy_val", 128, 4):
        cfg = gen_base_cfg("none", batch_size=2, num_classes=100)
        cfg = ClassificationDALIDataModule.add_and_assert_specific_cfg(cfg)

        backbone = resnet18()
        backbone.fc = nn.Identity()

        cfg.dali.device = "cpu"
        cfg.data.train_path = "./dummy_train"
        cfg.data.val_path = "./dummy_val"
        cfg.data.dataset = "custom"

        model = LinearModel(backbone, cfg=cfg)
        trainer = gen_trainer(cfg)

        dali_datamodule = ClassificationDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            data_fraction=-1,
            dali_device=cfg.dali.device,
        )

        trainer = gen_trainer(cfg)
        trainer.fit(model, datamodule=dali_datamodule)
