import math

import numpy as np
from PIL import Image
from solo.utils.classification_dataloader import prepare_data, prepare_datasets, prepare_transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def test_transforms():
    im = np.random.rand(32, 32, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")

    T_train, T_val = prepare_transforms("cifar10")
    assert T_train(im).size(1) == 32
    assert T_val(im).size(1) == 32

    T_train, T_val = prepare_transforms("cifar100")
    assert T_train(im).size(1) == 32
    assert T_val(im).size(1) == 32

    im = np.random.rand(500, 300, 3) * 255
    im = Image.fromarray(im.astype("uint8")).convert("RGB")

    T_train, T_val = prepare_transforms("stl10")
    assert T_train(im).size(1) == 96
    assert T_val(im).size(1) == 96

    T_train, T_val = prepare_transforms("imagenet100")
    assert T_train(im).size(1) == 224
    assert T_val(im).size(1) == 224


def test_datasets():
    T_train, T_val = prepare_transforms("cifar10")
    train_dataset, val_dataset = prepare_datasets("cifar10", T_train, T_val, data_dir="./datasets")
    assert isinstance(train_dataset, CIFAR10)
    assert isinstance(val_dataset, CIFAR10)
    assert len(train_dataset[0]) == 2
    assert len(val_dataset[0]) == 2


def test_data():
    bs = 64
    num_samples_train = 50000
    num_samples_val = 10000
    num_batches_train = num_samples_train // bs
    num_batches_val = math.ceil(num_samples_val / bs)

    train_loader, val_loader = prepare_data("cifar10", batch_size=bs, num_workers=0)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert num_batches_train == len(train_loader)
    assert num_batches_val == len(val_loader)
