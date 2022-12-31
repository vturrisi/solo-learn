# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
        for D in [self.train_dir, self.val_dir]:
            for y in range(self.num_classes):
                # make needed directories
                with contextlib.suppress(OSError):
                    os.makedirs(D / str(y))

                for i in range(self.size):
                    # generate random image
                    size = (random.randint(300, 400), random.randint(300, 400))
                    im = np.random.rand(*size, 3) * 255
                    im = Image.fromarray(im.astype("uint8")).convert("RGB")
                    im.save(D / str(y) / f"{i}.jpg")

    def __exit__(self, *args):
        with contextlib.suppress(OSError):
            shutil.rmtree(self.train_dir)
        with contextlib.suppress(OSError):
            shutil.rmtree(self.val_dir)
