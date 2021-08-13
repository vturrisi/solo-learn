import os
import random
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Type, Union

import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = [0.1, 2.0]):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gaussian blur to an input image.

        Args:
            x (torch.Tensor): an image in the tensor format.

        Returns:
            torch.Tensor: returns a blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)


class NCropAugmentation:
    def __init__(self, transform: Union[Callable, Sequence], num_crops: Optional[int] = None):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Union[Callable, Sequence]): transformation pipeline or list of
                transformation pipelines.
            num_crops: if transformation pipeline is not a list, applies the same
                pipeline num_crops times, if it is a list, this is ignored and each
                element of the list is applied once.
        """

        self.transform = transform

        if isinstance(transform, Iterable):
            self.one_transform_per_crop = True
            assert num_crops == len(transform)
        else:
            self.one_transform_per_crop = False
            self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        if self.one_transform_per_crop:
            return [transform(x) for transform in self.transform]
        else:
            return [self.transform(x) for _ in range(self.num_crops)]


class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
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
        min_scale: float = 0.08,
    ):
        """Applies cifar transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        """

        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (32, 32),
                    scale=(min_scale, 1.0),
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
        min_scale: float = 0.08,
    ):
        """Applies STL10 transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (96, 96),
                    scale=(min_scale, 1.0),
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
        size: int = 224,
        min_scale: float = 0.08,
    ):
        """Class that applies Imagenet transformations.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            size (int, optional): size of the crop. Defaults to 224.
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=(min_scale, 1.0),
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


class CustomTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        min_scale: float = 0.08,
        size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """Class that applies Custom transformations.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.0.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            size (int, optional): size of the crop. Defaults to 224.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to (0.485, 0.456, 0.406).
            std (Sequence[float], optional): std values for normalization.
                Defaults to (0.228, 0.224, 0.225).
        """

        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=(min_scale, 1.0),
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
                transforms.Normalize(mean=mean, std=std),
            ]
        )


class MulticropAugmentation:
    def __init__(
        self,
        transform: Callable,
        size_crops: Sequence[int],
        num_crops: Sequence[int],
        min_scales: Sequence[float],
        max_scale_crops: Sequence[float],
    ):
        """Class that applies multi crop augmentation.

        Args:
            transform (Callable): transformation callable without cropping.
            size_crops (Sequence[int]): a sequence of sizes of the crops.
            num_crops (Sequence[int]): a sequence number of crops per crop size.
            min_scales (Sequence[float]): sequence of minimum crop scales per crop
                size.
            max_scale_crops (Sequence[float]): sequence of maximum crop scales per crop
                size.
        """

        self.size_crops = size_crops
        self.num_crops = num_crops
        self.min_scales = min_scales
        self.max_scale_crops = max_scale_crops

        self.transforms = []
        for i in range(len(size_crops)):
            rrc = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scales[i], max_scale_crops[i]),
                interpolation=transforms.InterpolationMode.BICUBIC,
            )
            full_transform = transforms.Compose([rrc, transform])
            self.transforms.append(full_transform)

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies multi crop augmentations.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: a list of crops in the tensor format.
        """

        imgs = []
        for n, transform in zip(self.num_crops, self.transforms):
            imgs.extend([transform(x) for i in range(n)])
        return imgs


class MulticropCifarTransform(BaseTransform):
    def __init__(self):
        """Class that applies multicrop transform for CIFAR"""

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
        """Class that applies multicrop transform for STL10"""

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
        """Class that applies multicrop transform for Imagenet.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.5.
            solarization_prob (float, optional): minimum scale of the crops. Defaults to 0.0.
        """

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


class MulticropCustomTransform(BaseTransform):
    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.228, 0.224, 0.225),
    ):
        """Class that applies multicrop transform for Custom Datasets.
        If you want to do exoteric augmentations, you can just re-write this class.

        Args:
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.5.
            solarization_prob (float, optional): minimum scale of the crops. Defaults to 0.0.
            mean (Sequence[float], optional): mean values for normalization.
                Defaults to (0.485, 0.456, 0.406).
            std (Sequence[float], optional): std values for normalization.
                Defaults to (0.228, 0.224, 0.225).
        """

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
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def prepare_transform(dataset: str, multicrop: bool = False, **kwargs) -> Any:
    """Prepares transforms for a specific dataset. Optionally uses multi crop.

    Args:
        dataset (str): name of the dataset.
        multicrop (bool, optional): whether or not to use multi crop. Defaults to False.

    Returns:
        Any: a transformation for a specific dataset.
    """

    if dataset in ["cifar10", "cifar100"]:
        return CifarTransform(**kwargs) if not multicrop else MulticropCifarTransform()
    elif dataset == "stl10":
        return STLTransform(**kwargs) if not multicrop else MulticropSTLTransform()
    elif dataset in ["imagenet", "imagenet100"]:
        return (
            ImagenetTransform(**kwargs) if not multicrop else MulticropImagenetTransform(**kwargs)
        )
    elif dataset == "custom":
        return CustomTransform(**kwargs) if not multicrop else MulticropCustomTransform(**kwargs)


def prepare_n_crop_transform(
    transform: Callable, num_crops: Optional[int] = None
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transform (Callable): a transformation.
        num_crops (Optional[int], optional): number of crops. Defaults to None.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    return NCropAugmentation(transform, num_crops)


def prepare_multicrop_transform(
    transform: Callable,
    size_crops: Sequence[int],
    num_crops: Optional[Sequence[int]] = None,
    min_scales: Optional[Sequence[float]] = None,
    max_scale_crops: Optional[Sequence[float]] = None,
) -> MulticropAugmentation:
    """Prepares multicrop transformations by creating custom crops given the parameters.

    Args:
        transform (Callable): transformation callable without cropping.
        size_crops (Sequence[int]): a sequence of sizes of the crops.
        num_crops (Optional[Sequence[int]]): list of number of crops per crop size.
        min_scales (Optional[Sequence[float]]): sequence of minimum crop scales per crop
            size.
        max_scale_crops (Optional[Sequence[float]]): sequence of maximum crop scales per crop
            size.

    Returns:
        MulticropAugmentation: prepared augmentation pipeline that supports multicrop with
            different sizes.
    """

    if num_crops is None:
        num_crops = [2, 6]
    if min_scales is None:
        min_scales = [0.14, 0.05]
    if max_scale_crops is None:
        max_scale_crops = [1.0, 0.14]

    return MulticropAugmentation(
        transform,
        size_crops=size_crops,
        num_crops=num_crops,
        min_scales=min_scales,
        max_scale_crops=max_scale_crops,
    )


def prepare_datasets(
    dataset: str,
    transform: Callable,
    data_dir: Optional[Union[str, Path]] = None,
    train_dir: Optional[Union[str, Path]] = None,
    no_labels: Optional[Union[str, Path]] = False,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        data_dir (Optional[Union[str, Path]], optional): the directory to load data from.
            Defaults to None.
        train_dir (Optional[Union[str, Path]], optional): training data directory
            to be appended to data_dir. Defaults to None.
        no_labels (Optional[bool], optional): if the custom dataset has no labels.

    Returns:
        Dataset: the desired dataset with transformations.
    """

    if data_dir is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        data_dir = sandbox_folder / "datasets"

    if train_dir is None:
        train_dir = Path(f"{dataset}/train")
    else:
        train_dir = Path(train_dir)

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            data_dir / train_dir,
            train=True,
            download=True,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            data_dir / train_dir,
            split="train+unlabeled",
            download=True,
            transform=transform,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        train_dir = data_dir / train_dir
        train_dataset = dataset_with_index(ImageFolder)(train_dir, transform)

    elif dataset == "custom":
        train_dir = data_dir / train_dir

        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_with_index(dataset_class)(train_dir, transform)

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int = 64, num_workers: int = 4
) -> DataLoader:
    """Prepares the training dataloader for pretraining.

    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.

    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader
