import os

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder, STL10


def prepare_transforms(dataset, normalize=True):
    if dataset in ["cifar10", "cifar100"]:
        T_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        T_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
    elif dataset == "stl10":
        T_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )
        T_val = transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )
    elif dataset in ["imagenet", "imagenet100"]:
        T_train = [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        T_val = [
            transforms.Resize(256),  # resize shorter
            transforms.CenterCrop(224),  # take center crop
            transforms.ToTensor(),
        ]
        if normalize:
            norm_t = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225))
            T_train.append(norm_t)
            T_val.append(norm_t)
        T_train = transforms.Compose(T_train)
        T_val = transforms.Compose(T_val)

    return T_train, T_val


def prepare_datasets(dataset, T_train, T_val, data_folder=None, train_dir=None, val_dir=None):
    if data_folder is None:
        sandbox_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data_folder = os.path.join(sandbox_folder, "datasets")

    if train_dir is None:
        train_dir = f"{dataset}/train"
    if val_dir is None:
        val_dir = f"{dataset}/test"

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = DatasetClass(
            os.path.join(data_folder, train_dir),
            train=True,
            download=True,
            transform=T_train,
        )

        val_dataset = DatasetClass(
            os.path.join(data_folder, val_dir),
            train=False,
            download=True,
            transform=T_val,
        )

    elif dataset == "stl10":
        train_dataset = STL10(
            os.path.join(data_folder, train_dir),
            split="train",
            download=True,
            transform=T_train,
        )
        val_dataset = STL10(
            os.path.join(data_folder, val_dir),
            split="test",
            download=True,
            transform=T_val,
        )

    elif dataset == "imagenet":
        train_dir = os.path.join(data_folder, train_dir)
        val_dir = os.path.join(data_folder, val_dir)

        train_dataset = ImageFolder(train_dir, T_train)
        val_dataset = ImageFolder(val_dir, T_val)

    elif dataset == "imagenet100":
        train_dir = os.path.join(data_folder, train_dir)
        val_dir = os.path.join(data_folder, val_dir)

        train_dataset = ImageFolder(train_dir, T_train)
        val_dataset = ImageFolder(val_dir, T_val)

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
    data_folder=None,
    train_dir=None,
    val_dir=None,
    batch_size=64,
    num_workers=4,
    normalize=True,
):
    T_train, T_val = prepare_transforms(dataset, normalize=normalize)
    train_dataset, val_dataset = prepare_datasets(
        dataset,
        T_train,
        T_val,
        data_folder=data_folder,
        train_dir=train_dir,
        val_dir=val_dir,
    )
    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
