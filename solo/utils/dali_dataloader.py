from typing import Callable, Iterable, Union

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class Mux:
    """Implements mutex operation for dali in order to support probabilitic augmentations."""

    def __init__(self, prob: float):
        """
        Args:
            prob: probability value
        """

        self.to_bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, true_case, false_case):
        condition = self.to_bool(self.rng())
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case


class RandomGrayScaleConversion:
    """Converts image to greyscale with probability."""

    def __init__(self, prob: float = 0.2, device: str = "gpu"):
        """
        Args:
            prob: probability of conversion
            device: device on which the operation will be performed

        """

        self.mux = Mux(prob=prob)
        self.grayscale = ops.ColorSpaceConversion(
            device=device, image_type=types.RGB, output_type=types.GRAY
        )

    def __call__(self, images):
        out = self.grayscale(images)
        out = fn.cat(out, out, out, axis=2)
        return self.mux(true_case=out, false_case=images)


class RandomColorJitter:
    """Applies random color jittering with probability."""

    def __init__(
        self,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        prob: float = 0.8,
        device: str = "gpu",
    ):
        assert 0 <= hue <= 0.5

        """
        Args:
            brightness: sampled uniformly in [max(0, 1 - brightness), 1 + brightness]
            contrast: sampled uniformly in [max(0, 1 - contrast), 1 + contrast]
            saturation: sampled uniformly in [max(0, 1 - saturation), 1 + saturation]
            hue: sampled uniformly in [-hue, hue]
            prob: probability of applying jitter
            device: device on which the operation will be performed

        """

        self.mux = Mux(prob=prob)

        # look at torchvision docs to see how colorjitter samples stuff
        # for bright, cont and sat, it samples from [1-v, 1+v]
        # for hue, it samples from [-hue, hue]
        self.color = ops.ColorTwist(device=device)
        self.brightness = ops.random.Uniform(range=[max(0, 1 - brightness), 1 + brightness])
        self.contrast = ops.random.Uniform(range=[max(0, 1 - contrast), 1 + contrast])
        self.saturation = ops.random.Uniform(range=[max(0, 1 - saturation), 1 + saturation])
        # dali uses hue in degrees for some reason...
        hue = 360 * hue
        self.hue = ops.random.Uniform(range=[-hue, hue])

    def __call__(self, images):
        out = self.color(
            images,
            brightness=self.brightness(),
            contrast=self.contrast(),
            saturation=self.saturation(),
            hue=self.hue(),
        )
        return self.mux(true_case=out, false_case=images)


class RandomGaussianBlur:
    """Applies random gaussian blur with probability."""

    def __init__(self, prob: float = 0.5, window_size: int = 23, device: str = "gpu"):
        """
        Args:
            prob: probability of applying random gaussian blur
            window_size: window size for gaussian blur
            device: device on which the operation will be performed

        """

        self.mux = Mux(prob=prob)
        # gaussian blur
        self.gaussian_blur = ops.GaussianBlur(device=device, window_size=(window_size, window_size))
        self.sigma = ops.random.Uniform(range=[0, 1])

    def __call__(self, images):
        sigma = self.sigma() * 1.9 + 0.1
        out = self.gaussian_blur(images, sigma=sigma)
        return self.mux(true_case=out, false_case=images)


class RandomSolarize:
    """Applies random solarization with probability."""

    def __init__(self, threshold: int = 128, prob: float = 0.0):
        """
        Args:
            threshold: threshold for inversion
            prob: probability of solarization

        """

        self.mux = Mux(prob=prob)

        self.threshold = threshold

    def __call__(self, images):
        inverted_img = 255 - images
        mask = images >= self.threshold
        out = mask * inverted_img + (True ^ mask) * images
        return self.mux(true_case=out, false_case=images)


class NormalPipeline(Pipeline):
    """Loads images and applies validation / linear eval transformations with dali."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: str,
        validation: bool = False,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
    ):
        """Initializes the pipeline for validation or linear eval training.

        If validation is set to True then images will only be resized to 256px and center cropped
        to 224px, otherwise random resized crop, horizontal flip are applied. In both cases images
        are normalized.

        Args:
            data_path (str): directory that contains the data.
            batch_size (int): batch size.
            device (str): device on which the operation will be performed.
            validation (bool): whether it is validation or training. Defaults to False. Defaults to
                False.
            device_id (int): id of the device used to initialize the seed and for parent class.
                Defaults to 0.
            shard_id (int): id of the shard (chuck of samples). Defaults to 0.
            num_shards (int): total number of shards. Defaults to 1.
            num_threads (int): number of threads to run in parallel. Defaults to 4.
            seed (int): seed for random number generation. Defaults to 12.
        """

        seed += device_id
        super().__init__(batch_size, num_threads, device_id, seed)

        self.device = device
        self.validation = validation

        self.reader = ops.readers.File(
            file_root=data_path,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=True if not self.validation else False,
        )
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )

        # crop operations
        if self.validation:
            self.resize = ops.Resize(
                device=self.device,
                resize_shorter=256,
                interp_type=types.INTERP_CUBIC,
            )
            # center crop and normalize
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(224, 224),
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )
        else:
            self.resize = ops.RandomResizedCrop(
                device=self.device,
                size=224,
                random_area=(0.08, 1.0),
                interp_type=types.INTERP_CUBIC,
            )
            # normalize and horizontal flip
            self.cmn = ops.CropMirrorNormalize(
                device=self.device,
                dtype=types.FLOAT,
                output_layout=types.NCHW,
                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
            )

        self.coin05 = ops.random.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

    def define_graph(self):
        """Defines the computational graph for dali operations."""
        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        images = self.resize(images)

        if self.validation:
            # crop and normalize
            images = self.cmn(images)
        else:
            # normalize and maybe apply horizontal flip with 0.5 chance
            images = self.cmn(images, mirror=self.coin05())

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)
        return (images, labels)


class ImagenetTransform:
    def __init__(
        self,
        device: str,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
        gaussian_prob: float = 0.5,
        solarization_prob: float = 0.0,
        size: int = 224,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
    ):
        """Applies Imagenet transformations to a batch of images.

        Args:
            device (str): device on which the operations will be performed.
            brightness (float): sampled uniformly in [max(0, 1 - brightness), 1 + brightness].
            contrast (float): sampled uniformly in [max(0, 1 - contrast), 1 + contrast].
            saturation (float): sampled uniformly in [max(0, 1 - saturation), 1 + saturation].
            hue (float): sampled uniformly in [-hue, hue].
            gaussian_prob (float, optional): probability of applying gaussian blur. Defaults to 0.5.
            solarization_prob (float, optional): probability of applying solarization. Defaults
                to 0.0.
            size (int, optional): size of the side of the image after transformation. Defaults
                to 224.
            min_scale (float, optional): minimum scale of the crops. Defaults to 0.08.
            max_scale (float, optional): maximum scale of the crops. Defaults to 1.0.
        """

        # random crop
        self.random_crop = ops.RandomResizedCrop(
            device=device,
            size=size,
            random_area=(min_scale, max_scale),
            interp_type=types.INTERP_CUBIC,
        )

        # color jitter
        self.random_color_jitter = RandomColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            prob=0.8,
            device=device,
        )

        # grayscale conversion
        self.random_grayscale = RandomGrayScaleConversion(prob=0.2, device=device)

        # gaussian blur
        self.random_gaussian_blur = RandomGaussianBlur(prob=gaussian_prob, device=device)

        # solarization
        self.random_solarization = RandomSolarize(prob=solarization_prob)

        # normalize and horizontal flip
        self.cmn = ops.CropMirrorNormalize(
            device=device,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.228 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin05 = ops.random.CoinFlip(probability=0.5)

    def __call__(self, images):
        out = self.random_crop(images)
        out = self.random_color_jitter(out)
        out = self.random_grayscale(out)
        out = self.random_gaussian_blur(out)
        out = self.random_solarization(out)
        out = self.cmn(out, mirror=self.coin05())
        return out


class PretrainPipeline(Pipeline):
    """Loads images and applies pretrain transformations with dali."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: str,
        transform: Union[Callable, Iterable],
        n_crops: int = 2,
        random_shuffle: bool = True,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
    ):
        """
        Initializes the pipeline for pretraining.

        Args:
            data_path: directory that contains the data
            batch_size: batch size
            device: device on which the operation will be performed
            transform: a transformation or a sequence of transformations to be applied
            n_crops: number of crops
            random_shuffle: whether to randomly shuffle the samples
            device_id: id of the device used to initialize the seed and for parent class
            shard_id: id of the shard (chuck of samples)
            num_shards: total number of shards
            num_threads: number of threads to run in parallel
            seed: seed for random number generation
        """

        seed += device_id
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )

        self.device = device
        self.reader = ops.readers.File(
            file_root=data_path,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle,
        )
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

        self.n_crops = n_crops

        # transformations
        self.transform = transform

        if isinstance(transform, Iterable):
            self.one_transform_per_crop = True
        else:
            self.one_transform_per_crop = False
            self.n_crops = n_crops

    def define_graph(self):
        """Defines the computational graph for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        if self.one_transform_per_crop:
            crops = [transform(images) for transform in self.transform]
        else:
            crops = [self.transform(images) for i in range(self.n_crops)]

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (*crops, labels)


class MulticropPretrainPipeline(Pipeline):
    """Loads images and applies multicrop transformations with dali."""

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        device: str,
        transforms: Iterable,
        n_crops: Iterable[int],
        size_crops: Iterable[int],
        min_scale_crops: Iterable[float],
        max_scale_crops: Iterable[float],
        random_shuffle: bool = True,
        device_id: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        seed: int = 12,
    ):
        """
        Initializes the pipeline for pretraining with multicrop.

        Args:
            data_path: directory that contains the data
            batch_size: batch size
            device: device on which the operation will be performed
            transforms: a sequence of transformations to be applied
            n_crops: number of crops
            size_crops: sequence of crop sizes images will be resized to
            min_scale_crops: sequence of minimum scales for each crop
            max_scale_crops: sequence of maximum scales for each crop
            random_shuffle: whether to randomly shuffle the samples
            device_id: id of the device used to initialize the seed and for parent class
            shard_id: id of the shard (chuck of samples)
            num_shards: total number of shards
            num_threads: number of threads to run in parallel
            seed: seed for random number generation
        """

        seed += device_id
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )

        self.device = device
        self.reader = ops.readers.File(
            file_root=data_path,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle,
        )
        decoder_device = "mixed" if self.device == "gpu" else "cpu"
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.decoders.Image(
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
        )
        self.to_int64 = ops.Cast(dtype=types.INT64, device=device)

        self.n_crops = n_crops
        self.size_crops = size_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        assert len(transforms) == len(size_crops)
        self.transforms = transforms

    def define_graph(self):
        """Defines the computational graph for dali operations."""

        # read images from memory
        inputs, labels = self.reader(name="Reader")
        images = self.decode(inputs)

        # crop into large and small images
        crops = []
        for i, transform in enumerate(self.transforms):
            for _ in range(self.n_crops[i]):
                crop = transform(images)
                crops.append(crop)

        if self.device == "gpu":
            labels = labels.gpu()
        # PyTorch expects labels as INT64
        labels = self.to_int64(labels)

        return (*crops, labels)
