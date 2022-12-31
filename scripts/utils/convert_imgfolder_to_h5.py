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

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


def convert_imgfolder_to_h5(folder_path: str, h5_path: str):
    """Converts image folder to a h5 dataset.

    Args:
        folder_path (str): path to the image folder.
        h5_path (str): output path of the h5 file.
    """

    with h5py.File(h5_path, "w") as h5:
        classes = os.listdir(folder_path)
        for class_name in tqdm(classes, desc="Processing classes"):
            cur_folder = os.path.join(folder_path, class_name)
            class_group = h5.create_group(class_name)
            for i, img_name in enumerate(os.listdir(cur_folder)):
                with open(os.path.join(cur_folder, img_name), "rb") as fid_img:
                    binary_data = fid_img.read()
                data = np.frombuffer(binary_data, dtype="uint8")
                class_group.create_dataset(
                    img_name,
                    data=data,
                    shape=data.shape,
                    compression="gzip",
                    compression_opts=9,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    parser.add_argument("--h5_path", type=str, required=True)
    args = parser.parse_args()
    convert_imgfolder_to_h5(args.folder_path, args.h5_path)
