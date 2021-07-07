import contextlib
import os
import random
import shutil

import numpy as np
from PIL import Image
from solo.utils.dali_dataloader import ImagenetTransform


class DummyDataset:
    def __init__(self, train_dir, val_dir, size, n_classes):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.size = size
        self.n_classes = n_classes

    def __enter__(self):
        for dir in [self.train_dir, self.val_dir]:
            for y in range(self.n_classes):
                # make needed directories
                with contextlib.suppress(OSError):
                    os.makedirs(os.path.join(dir, str(y)))

                for i in range(self.size):
                    # generate random image
                    size = (random.randint(300, 400), random.randint(300, 400))
                    im = np.random.rand(*size, 3) * 255
                    im = Image.fromarray(im.astype("uint8")).convert("RGB")
                    im.save(os.path.join(dir, str(y), f"{i}.jpg"))

    def __exit__(self, *args):
        with contextlib.suppress(OSError):
            shutil.rmtree(self.train_dir)
        with contextlib.suppress(OSError):
            shutil.rmtree(self.val_dir)


def test_dali_dataloader():
    # creates a dummy dataset that autodeletes after usage
    with DummyDataset("dummy_train", "dummy_val", 10, 4):
        transform = ImagenetTransform(
            device="gpu",
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            gaussian_prob=0.5,
            solarization_prob=0.1,
            size=224,
            min_scale=0.08,
            max_scale=0.08,
        )
