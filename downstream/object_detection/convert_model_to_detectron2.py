# Copyright 2021 solo-learn development team.

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

from argparse import ArgumentParser
import pickle as pkl
import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained_feature_extractor", type=str, required=True)
    parser.add_argument("--output_detectron_model", type=str, required=True)

    args = parser.parse_args()

    checkpoint = torch.load(args.pretrained_feature_extractor, map_location="cpu")
    checkpoint = checkpoint["state_dict"]

    newmodel = {}
    for k, v in checkpoint.items():
        if not k.startswith("backbone"):
            continue

        old_k = k
        k = k.replace("backbone.", "")
        if "layer" not in k:
            k = "stem." + k
        for t in [1, 2, 3, 4]:
            k = k.replace(f"layer{t}", f"res{t + 1}")
        for t in [1, 2, 3]:
            k = k.replace(f"bn{t}", f"conv{t}.norm")
        k = k.replace("downsample.0", "shortcut")
        k = k.replace("downsample.1", "shortcut.norm")
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"model": newmodel, "__author__": "solo-learn", "matching_heuristics": True}

    with open(args.output_detectron_model, "wb") as f:
        pkl.dump(res, f)
