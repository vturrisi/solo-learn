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

from solo.utils.dali_dataloader import (
    PretrainPipeline,
    ImagenetTransform,
    MulticropPretrainPipeline,
    NormalPipeline,
)
from .utils import DummyDataset


def test_dali_dataloader():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("dummy_train", "dummy_val", 10, 4):
        num_crops = [2, 4]
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
                size=size,
                min_scale=min_scale,
                max_scale=max_scale,
            )
            transforms.append(transform)

        # multicrop pipeline
        train_pipeline = MulticropPretrainPipeline(
            "dummy_train",
            batch_size=4,
            transforms=transforms,
            num_crops=num_crops,
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        train_pipeline.build()

        # simple pretrain pipeline
        train_pipeline = PretrainPipeline(
            "dummy_train",
            batch_size=4,
            transform=transforms[0],
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        train_pipeline.build()

        # normal pipeline
        train_pipeline = NormalPipeline(
            "dummy_train",
            validation=False,
            batch_size=4,
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        train_pipeline.build()

        val_pipeline = NormalPipeline(
            "dummy_val",
            validation=True,
            batch_size=4,
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        val_pipeline.build()
