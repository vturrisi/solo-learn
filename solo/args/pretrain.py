import argparse
import os

import omegaconf
import pytorch_lightning as pl
from omegaconf import OmegaConf
from solo.data.dali_dataloader import PretrainDALIDataModule
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.auto_umap import AutoUMAP
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import omegaconf_select
from hydra._internal.utils import get_args

_N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
}

_SUPPORTED_DATASETS = [
    "cifar10",
    "cifar100",
    "stl10",
    "imagenet",
    "imagenet100",
    "custom",
]


def add_and_assert_dataset_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    assert not OmegaConf.is_missing(cfg, "data.dataset")
    assert not OmegaConf.is_missing(cfg, "data.train_path")
    assert not OmegaConf.is_missing(cfg, "data.format")

    assert cfg.data.dataset in _SUPPORTED_DATASETS

    # if validation path is not available, assume that we want to skip eval
    cfg.data.val_path = omegaconf_select(cfg, "data.val_path", None)
    cfg.data.no_labels = omegaconf_select(cfg, "data.no_labels", False)
    cfg.data.fraction = omegaconf_select(cfg, "data.fraction", -1)
    cfg.data.debug = omegaconf_select(cfg, "data.debug", False)

    return cfg


def parse_cfg(cfg: omegaconf.DictConfig):
    print(cfg)
    exit()
    parser = argparse.ArgumentParser()
    # add pytorch lightning trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--cfg", required=True)

    parser.add_argument(
        "rest",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # collect PL parameters
    pl_args = {k: v for k, v in vars(args).items() if k != "cfg"}
    cfg = OmegaConf.create(pl_args)

    # merge with cfg file
    # allows overwriting PL parameters and replaces cfg files with the actual configuration
    cfg_from_file = OmegaConf.load(args.cfg)
    for k, v in cfg_from_file.items():
        if k.endswith("_cfg"):
            aux_cfg_file = OmegaConf.load(v)
            cfg_from_file.merge_with(aux_cfg_file)
            del cfg_from_file[k]
    cfg.merge_with(cfg_from_file)

    # replace cfg with remaining parameters
    if args.rest:
        cfg.merge_with(OmegaConf.from_cli(args.rest))

    # extra processing
    if cfg.data.dataset in _N_CLASSES_PER_DATASET:
        cfg.data.num_classes = _N_CLASSES_PER_DATASET[cfg.data.dataset]
    else:
        # hack to maintain the current pipeline
        # even if the custom dataset doesn't have any labels
        cfg.data.num_classes = max(
            1,
            len([entry.name for entry in os.scandir(cfg.data.train_data_path) if entry.is_dir]),
        )

    # find number of big/small crops
    big_size = cfg.data.augmentations[0].crop_size
    num_large_crops = num_small_crops = 0
    for pipeline in cfg.data.augmentations:
        if big_size == pipeline.crop_size:
            num_large_crops += pipeline.num_crops
        else:
            num_small_crops += pipeline.num_crops
    cfg.data.num_large_crops = num_large_crops
    cfg.data.num_small_crops = num_small_crops

    if cfg.data.format == "dali":
        assert cfg.data.dataset in ["imagenet100", "imagenet", "custom"]

    # adjust lr according to batch size
    num_nodes = cfg.num_nodes or 1
    scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * num_nodes / 256
    cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
    cfg.optimizer.classifier_lr = cfg.optimizer.classifier_lr * scale_factor

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
        cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
            cfg,
            "optimizer.kwargs.exclude_bias_n_norm",
            False,
        )
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    # method specific cfg
    cfg.method_kwargs = omegaconf_select(cfg, "method_kwargs", {})
    cfg = METHODS[cfg.method].add_and_assert_specific_cfg(cfg)

    # default values for checkpointer
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)

    # default values for auto_resume
    cfg = AutoResumer.add_and_assert_specific_cfg(cfg)

    # default values for auto_umap
    cfg = AutoUMAP.add_and_assert_specific_cfg(cfg)

    # default values for dali
    cfg = PretrainDALIDataModule.add_and_assert_specific_cfg(cfg)

    # assert dataset parameters
    cfg = add_and_assert_dataset_cfg(cfg)

    # default values for wandb
    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)

    # default misc values
    cfg.seed = omegaconf_select(cfg, "seed", 5)

    return cfg
