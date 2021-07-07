import contextlib
import os
import random
import shutil

import numpy as np
from PIL import Image
from solo.utils.dali_dataloader import (
    ContrastivePipeline,
    ImagenetTransform,
    MulticropContrastivePipeline,
    NormalPipeline,
)


class DummyDataset:
    def __init__(self, train_dir, val_dir, size, n_classes):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.size = size
        self.n_classes = n_classes

    def __enter__(self):
        for dir in [self.train_dir, self.val_dir]:
            for y in range(self.n_classes):
                # make needed directories
                with contextlib.suppress(OSError):
                    os.makedirs(os.path.join(dir, str(y)))

                for i in range(self.size):
                    # generate random image
                    size = (random.randint(300, 400), random.randint(300, 400))
                    im = np.random.rand(*size, 3) * 255
                    im = Image.fromarray(im.astype("uint8")).convert("RGB")
                    im.save(os.path.join(dir, str(y), f"{i}.jpg"))

    def __exit__(self, *args):
        with contextlib.suppress(OSError):
            shutil.rmtree(self.train_dir)
        with contextlib.suppress(OSError):
            shutil.rmtree(self.val_dir)


def test_dali_dataloader():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("dummy_train", "dummy_val", 10, 4):
        n_crops = [2, 4]
        size_crops = [224, 96]
        min_scale_crops = [0.14, 0.05]
        max_scale_crops = [1.0, 0.14]

        transforms = []
        for size, min_scale, max_scale in zip(size_crops, min_scale_crops, max_scale_crops):
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
        train_pipeline = MulticropContrastivePipeline(
            "dummy_train",
            batch_size=4,
            transforms=transforms,
            n_crops=n_crops,
            size_crops=size_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            device="cpu",
            device_id=0,
            shard_id=0,
            num_shards=1,
            num_threads=1,
        )
        train_pipeline.build()

        # output_map = [
        #     *[f"large{i}" for i in range(n_crops[0])],
        #     *[f"small{i}" for i in range(n_crops[1])],
        #     "label",
        # ]
        # policy = LastBatchPolicy.DROP
        # ContrastiveWrapper(
        #     train_pipeline,
        #     output_map=output_map,
        #     reader_name="Reader",
        #     last_batch_policy=policy,
        #     auto_reset=True,
        #     model_batch_size=4,
        #     model_rank=0,
        #     model_device="cpu",
        # )

        # simple contrastive pipeline
        train_pipeline = ContrastivePipeline(
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

        # output_map = ["large1", "large2", "label"]
        # policy = LastBatchPolicy.DROP
        # train_loader = ContrastiveWrapper(
        #     train_pipeline,
        #     output_map=output_map,
        #     reader_name="Reader",
        #     last_batch_policy=policy,
        #     auto_reset=True,
        #     model_batch_size=4,
        #     model_rank=0,
        #     model_device="cpu",
        # )

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
