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
