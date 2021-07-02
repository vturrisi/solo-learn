import os
import random
from typing import Callable, Iterable, List, Optional, Tuple, Union

from PIL import ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder


def dataset_with_index(DatasetClass):
    """
    Factory for datasets that also returns the data index.

    Args:
        DatasetClass: Dataset class to be wrapped

    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class GaussianBlur:
    def __init__(self, sigma: Union[List[float], Tuple[float]] = [0.1, 2.0]):
        """
        Gaussian blur as a callable object.

        Args:
            sigma: range to sample the radius of the guassian blur filter

        """

        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """
    Solarization as a callable object.

    """

    def __call__(self, img):
        return ImageOps.solarize(img)


class NCropAugmentation:
    def __init__(self, transform: Union[Callable, List, Tuple], n_crops: Optional[int] = None):
        """
        Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform: transformation pipeline or list of transformation pipelines
            n_crops: if transformation pipeline is not a list, applies the same
                pipeline n_crops times, if it is a list, this is ignored and each
                element of the list is applied once

        """

        self.transform = transform

        if isinstance(transform, Iterable):
            self.one_transform_per_crop = True
            assert n_crops == len(transform)
        else:
            self.one_transform_per_crop = False
            self.n_crops = n_crops

    def __call__(self, x):
        if self.one_transform_per_crop:
            return [transform(x) for transform in self.transform]
        else:
            return [self.transform(x) for _ in range(self.n_crops)]


class BaseTransform:
    """
    Adds callable base class to implement different transformation pipelines.

    """

    def __call__(self, x):
        return self.transform(x)

    def __repr__(self):
        return str(self.transform)


class CifarTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale_crop: float = 0.08,
    ):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (32, 32),
                    scale=(min_scale_crop, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )


class STLTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.0,
        solarization_prob: float = 0.0,
        min_scale_crop: float = 0.08,
    ):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (96, 96),
                    scale=(min_scale_crop, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )


class ImagenetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale_crop: float = 0.08,
    ):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(min_scale_crop, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )


class MulticropAugmentation:
    def __init__(
        self,
        transform: Callable,
        size_crops: Union[List[int], Tuple[int]],
        n_crops: Union[List[int], Tuple[int]],
        min_scale_crops: Union[List[float], Tuple[float]],
        max_scale_crops: Union[List[float], Tuple[float]],
    ):
        self.size_crops = size_crops
        self.n_crops = n_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        self.transforms = []
        for i in range(len(size_crops)):
            rrc = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
            full_transform = transforms.Compose([rrc, transform])
            self.transforms.append(full_transform)

    def __call__(self, x):
        imgs = []
        for n, transform in zip(self.n_crops, self.transforms):
            imgs.extend([transform(x) for i in range(n)])
        return imgs


class MulticropCifarTransform(BaseTransform):
    def __init__(self):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )


class MulticropSTLTransform(BaseTransform):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            ]
        )


class MulticropImagenetTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
    ):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness, contrast, saturation, hue)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=gaussian_prob),
                transforms.RandomApply([Solarization()], p=solarization_prob),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
            ]
        )


def prepare_transform(dataset, multicrop: bool = False, **kwargs):
    if dataset in ["cifar10", "cifar100"]:
        return CifarTransform(**kwargs) if not multicrop else MulticropCifarTransform()
    elif dataset == "stl10":
        return STLTransform(**kwargs) if not multicrop else MulticropSTLTransform()
    elif dataset in ["imagenet", "imagenet100"]:
        return (
            ImagenetTransform(**kwargs) if not multicrop else MulticropImagenetTransform(**kwargs)
        )


def prepare_n_crop_transform(transform: Callable, n_crops: Optional[int] = None):
    return NCropAugmentation(transform, n_crops)


def prepare_multicrop_transform(
    transform: Callable,
    size_crops: Union[List[int], Tuple[int]],
    n_crops: Optional[Union[List[int], Tuple[int]]] = None,
    min_scale_crops: Optional[Union[List[float], Tuple[float]]] = None,
    max_scale_crops: Optional[Union[List[float], Tuple[float]]] = None,
):
    if n_crops is None:
        n_crops = [2, 6]
    if min_scale_crops is None:
        min_scale_crops = [0.14, 0.05]
    if max_scale_crops is None:
        max_scale_crops = [1.0, 0.14]

    return MulticropAugmentation(
        transform,
        size_crops=size_crops,
        n_crops=n_crops,
        min_scale_crops=min_scale_crops,
        max_scale_crops=max_scale_crops,
    )


def prepare_datasets(
    dataset,
    transform: Callable,
    data_dir: Optional[str] = None,
    train_dir: Optional[str] = None,
):
    if data_dir is None:
        sandbox_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        data_dir = os.path.join(sandbox_folder, "datasets")

    if train_dir is None:
        train_dir = f"{dataset}/train"

    if dataset == "cifar10":
        train_dataset = dataset_with_index(CIFAR10)(
            os.path.join(data_dir, train_dir),
            train=True,
            download=True,
            transform=transform,
        )

    elif dataset == "cifar100":
        train_dataset = dataset_with_index(CIFAR100)(
            os.path.join(data_dir, train_dir),
            train=True,
            download=True,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            os.path.join(data_dir, train_dir),
            split="train+unlabeled",
            download=True,
            transform=transform,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        train_dir = os.path.join(data_dir, train_dir)
        train_dataset = dataset_with_index(ImageFolder)(train_dir, transform)

    return train_dataset


def prepare_dataloaders(train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
