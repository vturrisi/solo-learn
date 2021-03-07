import math
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder


class MultiCropAugmentation:
    def __init__(
        self, T, size_crops, nmb_crops, min_scale_crops, max_scale_crops,
    ):
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([randomresizedcrop, *T])] * nmb_crops[i])
        self.transform = trans

    def __call__(self, x):
        # this preserves the class label
        if isinstance(x, Image.Image):
            multi_crops = [trans(x) for trans in self.transform]
            return multi_crops
        else:
            return x


class MultiCropAugmentationConsensus:
    def __init__(
        self, T, size_crops, nmb_crops, min_scale_crops, max_scale_crops,
    ):
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        # i will assume that we are always normalizing,
        # so i can just take the last two transforms out
        self.T_tensor = transforms.Compose(T[-2:])
        T = T[:-2]

        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([randomresizedcrop, *T])])
        self.transform = trans

    def __call__(self, x):
        # this preserves the class label
        if isinstance(x, Image.Image):
            large_crops = [self.transform[0](x) for i in range(self.nmb_crops[0])]
            small_crops = [self.transform[1](x) for i in range(self.nmb_crops[1])]
            multi_crops = [
                *[self.T_tensor(crop) for crop in large_crops],
                *[self.T_tensor(crop) for crop in small_crops],
            ]
            return multi_crops
        else:
            return x


class NAugmentationsTransform:
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        # this preserves the class label
        if isinstance(x, Image.Image):
            return [self.transform(x) for i in range(self.n)]
        else:
            return x


class ImageFolderWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return index, (*data)


class RandomGaussianBlur:
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return Image.fromarray(cv2.GaussianBlur(np.asarray(img), (23, 23), sigma))


def get_color_distortion(strength=1.0):
    color_jitter = transforms.ColorJitter(
        0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength
    )
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def prepare_transformations(dataset, n_augs=2):
    if dataset in ["cifar10", "cifar100"]:
        T = transforms.Compose(
            [
                transforms.RandomResizedCrop((32, 32), scale=(0.2, 1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
    elif dataset == "stl10":
        T = transforms.Compose(
            [
                transforms.RandomResizedCrop((96, 96), scale=(0.2, 1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )
    elif dataset in ["imagenet", "imagenet100"]:
        T = [
            transforms.RandomResizedCrop((224, 224), scale=(0.2, 1)),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            RandomGaussianBlur(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
        ]
        T = transforms.Compose(T)

    T = NAugmentationsTransform(T, n_augs)
    return T


def prepare_transformations_multicrop(dataset, nmb_crops=None, consensus=False):
    if nmb_crops is None:
        nmb_crops = [2, 6]
    min_scale_crops = [0.14, 0.05]
    max_scale_crops = [1.0, 0.14]
    if dataset in ["cifar10", "cifar100"]:
        size_crops = [32, 24]
        T = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    elif dataset == "stl10":
        size_crops = [96, 58]
        T = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    elif dataset in ["imagenet", "imagenet100"]:
        size_crops = [224, 96]
        T = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Compose([get_color_distortion(strength=1), RandomGaussianBlur()]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
        ]
    if consensus:
        T = MultiCropAugmentationConsensus(
            T,
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
        )
    else:
        T = MultiCropAugmentation(
            T,
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
        )
    return T


def prepare_datasets(
    dataset,
    T,
    data_folder=None,
    train_dir=None,
    val_dir=None,
    pseudo_labels_path=None,
    with_index=False,
):
    if data_folder is None:
        if os.path.isdir("/data/datasets"):
            data_folder = "/data/datasets"
        elif os.path.isdir("/datasets"):
            data_folder = "/datasets/"
        else:
            sandbox_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            data_folder = os.path.join(sandbox_folder, "datasets")

    if train_dir is None:
        train_dir = f"{dataset}/train"

    if val_dir is None:
        val_dir = f"{dataset}/test"

    if dataset == "cifar10":
        train_dataset = CIFAR10(
            os.path.join(data_folder, train_dir), train=True, download=True, transform=T,
        )

        val_dataset = CIFAR10(
            os.path.join(data_folder, val_dir), train=False, download=True, transform=T,
        )

    elif dataset == "cifar100":
        train_dataset = CIFAR100(
            os.path.join(data_folder, train_dir), train=True, download=True, transform=T,
        )

        val_dataset = CIFAR100(
            os.path.join(data_folder, val_dir), train=False, download=True, transform=T,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            os.path.join(data_folder, train_dir),
            split="train+unlabeled",
            download=True,
            transform=T,
        )
        val_dataset = STL10(
            os.path.join(data_folder, val_dir), split="test", download=True, transform=T,
        )

    elif dataset == "imagenet":
        train_dir = os.path.join(data_folder, train_dir)
        val_dir = os.path.join(data_folder, val_dir)

        if pseudo_labels_path is not None:
            train_dataset = DatasetWithPseudoLabels(train_dir, T, pseudo_labels_path)
        else:
            train_dataset = ImageFolder(train_dir, T)

        val_dataset = ImageFolder(val_dir, T)

    elif dataset == "imagenet100":
        train_dir = os.path.join(data_folder, train_dir)
        val_dir = os.path.join(data_folder, val_dir)

        if pseudo_labels_path is not None:
            train_dataset = DatasetWithPseudoLabels(train_dir, T, pseudo_labels_path)
        else:
            train_dataset = ImageFolder(train_dir, T)

        val_dataset = ImageFolder(val_dir, T)

    if with_index:
        train_dataset = ImageFolderWithIndex(train_dataset)
        val_dataset = ImageFolderWithIndex(val_dataset)

    return train_dataset, val_dataset


def prepare_dataloaders(train_dataset, val_dataset, n_augs=2, batch_size=64, num_workers=4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def prepare_data(
    dataset,
    n_augs,
    data_folder=None,
    train_dir=None,
    val_dir=None,
    batch_size=64,
    num_workers=4,
    pseudo_labels_path=None,
    with_index=False,
):
    T = prepare_transformations(dataset, n_augs=n_augs)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T,
        data_folder=data_folder,
        train_dir=train_dir,
        val_dir=val_dir,
        pseudo_labels_path=pseudo_labels_path,
        with_index=with_index,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        n_augs=n_augs,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def prepare_data_multicrop(
    dataset,
    nmb_crops=None,
    consensus=False,
    data_folder=None,
    train_dir=None,
    val_dir=None,
    batch_size=64,
    num_workers=4,
    pseudo_labels_path=None,
    with_index=False,
):
    T = prepare_transformations_multicrop(dataset, nmb_crops=nmb_crops, consensus=consensus)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T,
        data_folder=data_folder,
        train_dir=train_dir,
        val_dir=val_dir,
        pseudo_labels_path=pseudo_labels_path,
        with_index=with_index,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset, val_dataset, batch_size=batch_size, num_workers=num_workers,
    )
    return train_loader, val_loader
