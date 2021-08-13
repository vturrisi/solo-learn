import argparse
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from pytorch_lightning import Trainer
from solo.methods import BarlowTwins
from solo.methods.dali import ClassificationABC, PretrainABC, NormalPipeline, Wrapper
from solo.methods.linear import LinearModel
from torch import nn
from torchvision.models.resnet import resnet18

from ..methods.utils import DATA_KWARGS, gen_base_kwargs
from .utils import DummyDataset


def test_dali_pretrain():
    for encode_indexes_into_labels in [True, False]:
        # creates a dummy dataset that autodeletes after usage
        with DummyDataset("dummy_train", "dummy_val", 128, 4):
            method_kwargs = {
                "proj_hidden_dim": 2048,
                "output_dim": 2048,
                "lamb": 5e-3,
                "scale_loss": 0.025,
                "encode_indexes_into_labels": encode_indexes_into_labels,
            }
            BASE_KWARGS = gen_base_kwargs(cifar=False, multicrop=False)
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
            )
            kwargs["unique_augs"] = 1

            MethodClass = type(f"Dali{BarlowTwins.__name__}", (BarlowTwins, PretrainABC), {})
            model = MethodClass(**kwargs)

            args = argparse.Namespace(**kwargs)
            trainer = Trainer.from_argparse_args(
                args,
                checkpoint_callback=False,
                limit_train_batches=2,
                limit_val_batches=2,
            )

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
            val_dl = Wrapper(
                val_pipeline,
                output_map=["x", "label"],
                reader_name="Reader",
                last_batch_policy=LastBatchPolicy.PARTIAL,
                auto_reset=True,
            )
            trainer.fit(model, val_dataloaders=val_dl)


def test_dali_linear():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("dummy_train", "dummy_val", 128, 4):
        BASE_KWARGS = gen_base_kwargs(cifar=False, momentum=True, multicrop=False)
        kwargs = {**BASE_KWARGS, **DATA_KWARGS}
        backbone = resnet18()
        backbone.fc = nn.Identity()

        kwargs["dali_device"] = "cpu"
        kwargs["data_dir"] = "."
        kwargs["train_dir"] = "dummy_train"
        kwargs["val_dir"] = "dummy_val"
        kwargs["dataset"] = "custom"

        MethodClass = type(f"Dali{LinearModel.__name__}", (LinearModel, ClassificationABC), {})
        model = MethodClass(backbone, **kwargs)

        args = argparse.Namespace(**kwargs)
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=False,
            limit_train_batches=2,
            limit_val_batches=2,
        )

        trainer.fit(model)
