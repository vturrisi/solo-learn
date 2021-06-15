import numpy as np
import torch
from PIL import Image
from solo.utils.contrastive_dataloader import prepare_n_crop_transform, prepare_transform


DATA_KWARGS = {
    "brightness": 0.4,
    "contrast": 0.4,
    "saturation": 0.2,
    "hue": 0.1,
    "gaussian_prob": 0.5,
    "solarization_prob": 0.5,
}


def gen_base_kwargs(cifar=False, momentum=False):
    BASE_KWARGS = {
        "encoder": "resnet18",
        "n_classes": 10,
        "cifar": cifar,
        "zero_init_residual": True,
        "max_epochs": 10,
        "optimizer": "sgd",
        "lars": True,
        "lr": 0.3,
        "weight_decay": 1e-10,
        "classifier_lr": 0.5,
        "exclude_bias_n_norm": True,
        "accumulate_grad_batches": 1,
        "extra_optimizer_args": {"momentum": 0.9},
        "scheduler": "warmup_cosine",
        "min_lr": 0.0,
        "warmup_start_lr": 0.0,
        "multicrop": False,
        "n_crops": 2,
        "n_small_crops": 0,
        "lr_decay_steps": None,
        "dali_device": "gpu",
        "asymmetric_augmentations": False,
        "last_batch_fill": False,
        "batch_size": 32,
        "num_workers": 4,
        "data_folder": "/data/datasets",
        "train_dir": "cifar10/train",
        "val_dir": "cifar10/val",
    }
    if momentum:
        BASE_KWARGS["base_tau_momentum"] = 0.99
        BASE_KWARGS["final_tau_momentum"] = 1.0
    return BASE_KWARGS


def gen_batch(b, n_classes, dataset):
    assert dataset in ["cifar10", "imagenet100"]

    if dataset == "cifar10":
        size = 32
    else:
        size = 224

    im = np.random.rand(size, size, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")
    T = prepare_transform(dataset, multicrop=False, **DATA_KWARGS)
    T = prepare_n_crop_transform(T, n_crops=2)
    x1, x2 = T(im)
    x1 = x1.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)
    x2 = x2.unsqueeze(0).repeat(b, 1, 1, 1).requires_grad_(True)

    idx = torch.arange(b)
    label = torch.randint(low=0, high=n_classes, size=(b,))

    batch, batch_idx = [idx, (x1, x2), label], 1

    return batch, batch_idx
