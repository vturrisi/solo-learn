import argparse
import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from solo.methods import METHODS

N_CLASSES_PER_DATASET = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "imagenet": 1000,
    "imagenet100": 100,
}


def parse():
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

    # extra processing
    if cfg.data.dataset in N_CLASSES_PER_DATASET:
        cfg.data.num_classes = N_CLASSES_PER_DATASET[cfg.data.dataset]
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
    cfg.optimizer.kwargs = cfg.get("optimizer.kwargs", {})
    if cfg.optimizer.name == "sgd":
        cfg.optimizer.kwargs.momentum = cfg.get("optimizer.kwargs.momentum", 0.9)
    elif cfg.optimizer.name == "lars":
        cfg.optimizer.kwargs.momentum = cfg.get("optimizer.kwargs.momentum", 0.9)
        cfg.optimizer.kwargs.eta = cfg.get("optimizer.kwargs.eta", 1e-3)
        cfg.optimizer.kwargs.grad_clip = cfg.get("optimizer.kwargs.grad_clip", False)
        cfg.optimizer.kwargs.exclude_bias_n_norm = cfg.get(
            "optimizer.kwargs.exclude_bias_n_norm", False
        )
    elif cfg.optimizer.name == "adamw":
        cfg.optimizer.kwargs.betas = cfg.get("optimizer.kwargs.betas", [0.9, 0.999])

    # method specific cfg
    cfg.method_kwargs = cfg.get("method_kwargs", {})
    cfg = METHODS[cfg.method].add_method_specific_cfg(cfg)

    # default values for checkpointer
    cfg.checkpoint = cfg.get("checkpoint", {})
    cfg.checkpoint.enabled = cfg.get("checkpoint.enabled", False)
    cfg.checkpoint.dir = cfg.get("checkpoint.dir", "trained_models")
    cfg.checkpoint.frequency = cfg.get("checkpoint.frequency", 1)

    # default values for auto_resume
    cfg.auto_resume = cfg.get("auto_resume", {})
    cfg.auto_resume.enabled = cfg.get("auto_resume.enabled", False)
    cfg.auto_resume.max_hours = cfg.get("auto_resume.max_hours", 36)

    # default values for auto_umap
    cfg.auto_umap = cfg.get("auto_umap", {})
    cfg.auto_umap.enabled = cfg.get("auto_umap.enabled", False)
    cfg.auto_umap.dir = cfg.get("auto_umap.dir", "auto_umap")
    cfg.auto_umap.frequency = cfg.get("auto_umap.frequency", 1)

    # default values for wandb
    cfg.wandb = cfg.get("wandb", {})
    cfg.wandb.enabled = cfg.get("wandb.enabled", False)
    cfg.wandb.project = cfg.get("wandb.project", "solo-learn")
    cfg.wandb.entity = cfg.get("wandb.entity", None)

    return cfg
