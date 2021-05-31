import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class Mux:
    # DALI doesn't support probabilistic augmentations, so we use muxing.
    def __init__(self, prob):
        self.to_bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, true_case, false_case):
        condition = self.to_bool(self.rng())
        neg_condition = condition ^ True
        return condition * true_case + neg_condition * false_case


class RandomGrayScaleConversion:
    def __init__(self, prob=0.2, device="gpu"):
        self.mux = Mux(prob=prob)
        self.grayscale = ops.ColorSpaceConversion(
            device=device, image_type=types.RGB, output_type=types.GRAY
        )

    def __call__(self, images):
        out = self.grayscale(images)
        out = fn.cat(out, out, out, axis=2)
        return self.mux(true_case=out, false_case=images)


class RandomColorJitter:
    def __init__(
        self, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, prob=0.8, device="gpu",
    ):
        assert 0 <= hue <= 0.5

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
    def __init__(self, prob=0.5, device="gpu"):
        self.mux = Mux(prob=prob)
        # gaussian blur
        self.gaussian_blur = ops.GaussianBlur(device=device, window_size=(23, 23))
        self.sigma = ops.random.Uniform(range=[0, 1])

    def __call__(self, images):
        sigma = self.sigma() * 1.9 + 0.1
        out = self.gaussian_blur(images, sigma=sigma)
        return self.mux(true_case=out, false_case=images)


class RandomSolarize:
    def __init__(self, threshold=128, prob=0.0):
        self.mux = Mux(prob=prob)

        self.threshold = threshold

    def __call__(self, images):
        inverted_img = 255 - images
        mask = images >= self.threshold
        out = mask * inverted_img + (True ^ mask) * images
        return self.mux(true_case=out, false_case=images)


class NormalPipeline(Pipeline):
    def __init__(
        self,
        data_path,
        batch_size,
        device,
        validation=False,
        device_id=0,
        shard_id=0,
        num_shards=1,
        num_threads=4,
        seed=12,
    ):
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
                device=self.device, resize_shorter=256, interp_type=types.INTERP_CUBIC,
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
        device,
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2,
        gaussian_prob=0.5,
        solarization_prob=0.0,
        size=224,
        min_scale=0.08,
        max_scale=1.0,
    ):
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


class ContrastivePipeline(Pipeline):
    def __init__(
        self,
        data_path,
        batch_size,
        device,
        transform,
        n_crops=2,
        random_shuffle=True,
        device_id=0,
        shard_id=0,
        num_shards=1,
        num_threads=4,
        seed=12,
    ):
        seed += device_id
        super().__init__(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed,
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

        if isinstance(transform, list):
            self.one_transform_per_crop = True
        else:
            self.one_transform_per_crop = False
            self.n_crops = n_crops

    def define_graph(self):
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


class MulticropContrastivePipeline(Pipeline):
    def __init__(
        self,
        data_path,
        batch_size,
        device,
        transforms,
        n_crops,
        size_crops,
        min_scale_crops,
        max_scale_crops,
        random_shuffle=True,
        device_id=0,
        shard_id=0,
        num_shards=1,
        num_threads=4,
        seed=12,
    ):
        seed += device_id
        super().__init__(
            batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed,
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

        assert isinstance(transforms, list) and len(transforms) == len(size_crops)

        self.transforms = transforms

    def define_graph(self):
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
