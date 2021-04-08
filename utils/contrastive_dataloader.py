import os

import torch
import torch.jit as jit
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from torchvision.io import read_image


class MultiCropAugmentation:
    def __init__(
        self,
        T,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        jit_transforms=False,
    ):
        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops
        self.jit_transforms = jit_transforms

        self.transforms = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )

            if jit_transforms:
                transform = jit.script(nn.Sequential(randomresizedcrop, *T))
            else:
                transform = transforms.Compose([randomresizedcrop, *T])
            self.transforms.append(transform)

    def __call__(self, x):
        imgs = []
        for n, transform in zip(self.nmb_crops, self.transforms):
            imgs.extend([transform(x) for i in range(n)])
        return imgs


class NAugmentationsTransform:
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        return [self.transform(x) for i in range(self.n)]


class ImageFolderWithIndex:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        return index, (*data)


def prepare_transformations(
    dataset, n_augs=2, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, jit_transforms=False
):
    if dataset in ["cifar10", "cifar100"]:
        T = transforms.Compose(
            [
                transforms.RandomResizedCrop((32, 32), scale=(0.2, 1.0)),
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
                transforms.RandomResizedCrop((96, 96), scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )
    elif dataset in ["imagenet", "imagenet100"]:
        T = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                nn.ModuleList([transforms.ColorJitter(brightness, contrast, saturation, hue)]),
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(23)]), p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ConvertImageDtype(torch.float) if jit_transforms else transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
        ]

        if jit_transforms:
            T = jit.script(nn.Sequential(*T))
        else:
            T = transforms.Compose(T)

    T = NAugmentationsTransform(T, n_augs)
    return T


def prepare_transformations_multicrop(
    dataset,
    brightness=0.8,
    contrast=0.8,
    saturation=0.8,
    hue=0.2,
    nmb_crops=None,
    jit_transforms=False,
):
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
            transforms.RandomApply(
                nn.ModuleList([transforms.ColorJitter(brightness, contrast, saturation, hue)]),
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(23)]), p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ConvertImageDtype(torch.float) if jit_transforms else transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
        ]
    T = MultiCropAugmentation(
        T,
        size_crops=size_crops,
        nmb_crops=nmb_crops,
        min_scale_crops=min_scale_crops,
        max_scale_crops=max_scale_crops,
        jit_transforms=jit_transforms,
    )
    return T


def prepare_datasets(
    dataset,
    T,
    data_folder=None,
    train_dir=None,
    val_dir=None,
    with_index=True,
    jit_transforms=False,
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
            os.path.join(data_folder, train_dir),
            train=True,
            download=True,
            transform=T,
        )

        val_dataset = CIFAR10(
            os.path.join(data_folder, val_dir),
            train=False,
            download=True,
            transform=T,
        )

    elif dataset == "cifar100":
        train_dataset = CIFAR100(
            os.path.join(data_folder, train_dir),
            train=True,
            download=True,
            transform=T,
        )

        val_dataset = CIFAR100(
            os.path.join(data_folder, val_dir),
            train=False,
            download=True,
            transform=T,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            os.path.join(data_folder, train_dir),
            split="train+unlabeled",
            download=True,
            transform=T,
        )
        val_dataset = STL10(
            os.path.join(data_folder, val_dir),
            split="test",
            download=True,
            transform=T,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        train_dir = os.path.join(data_folder, train_dir)
        val_dir = os.path.join(data_folder, val_dir)

        if jit_transforms:
            train_dataset = ImageFolder(train_dir, T, loader=read_image)
            val_dataset = ImageFolder(val_dir, T, loader=read_image)
        else:
            train_dataset = ImageFolder(train_dir, T)
            val_dataset = ImageFolder(val_dir, T)

    if with_index:
        train_dataset = ImageFolderWithIndex(train_dataset)
        val_dataset = ImageFolderWithIndex(val_dataset)

    return train_dataset, val_dataset


def prepare_dataloaders(train_dataset, val_dataset, batch_size=64, num_workers=4):
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
    n_augs=2,
    brightness=0.8,
    contrast=0.8,
    saturation=0.8,
    hue=0.2,
    data_folder=None,
    train_dir=None,
    val_dir=None,
    batch_size=64,
    num_workers=4,
    with_index=True,
    jit_transforms=False,
):
    T = prepare_transformations(
        dataset,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        n_augs=n_augs,
        jit_transforms=jit_transforms,
    )
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T,
        data_folder=data_folder,
        train_dir=train_dir,
        val_dir=val_dir,
        with_index=with_index,
        jit_transforms=jit_transforms,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def prepare_data_multicrop(
    dataset,
    brightness=0.8,
    contrast=0.8,
    saturation=0.8,
    hue=0.2,
    nmb_crops=None,
    data_folder=None,
    train_dir=None,
    val_dir=None,
    batch_size=64,
    num_workers=4,
    with_index=True,
    jit_transforms=False,
):
    T = prepare_transformations_multicrop(
        dataset,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        nmb_crops=nmb_crops,
        jit_transforms=jit_transforms,
    )
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T,
        data_folder=data_folder,
        train_dir=train_dir,
        val_dir=val_dir,
        with_index=with_index,
        jit_transforms=jit_transforms,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
