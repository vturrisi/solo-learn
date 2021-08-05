import contextlib
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


class DummyDataset:
    def __init__(self, train_dir, val_dir, size, num_classes):
        self.train_dir = Path(train_dir)
        self.val_dir = Path(val_dir)
        self.size = size
        self.num_classes = num_classes

    def __enter__(self):
        for dir in [self.train_dir, self.val_dir]:
            for y in range(self.num_classes):
                # make needed directories
                with contextlib.suppress(OSError):
                    os.makedirs(dir / str(y))

                for i in range(self.size):
                    # generate random image
                    size = (random.randint(300, 400), random.randint(300, 400))
                    im = np.random.rand(*size, 3) * 255
                    im = Image.fromarray(im.astype("uint8")).convert("RGB")
                    im.save(dir / str(y) / f"{i}.jpg")

    def __exit__(self, *args):
        with contextlib.suppress(OSError):
            shutil.rmtree(self.train_dir)
        with contextlib.suppress(OSError):
            shutil.rmtree(self.val_dir)
