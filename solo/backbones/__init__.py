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


from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
from .poolformer import (
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    poolformer_m36,
    poolformer_m48,
)
from .resnet import resnet18, resnet50
from .swin import swin_tiny, swin_small, swin_base, swin_large
from .vit import vit_tiny, vit_small, vit_base, vit_large
from .wide_resnet import wide_resnet28w2, wide_resnet28w8

__all__ = [
    "resnet18",
    "resnet50",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "swin_tiny",
    "swin_small",
    "swin_base",
    "swin_large",
    "poolformer_s12",
    "poolformer_s24",
    "poolformer_s36",
    "poolformer_m36",
    "poolformer_m48",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "wide_resnet28w2",
    "wide_resnet28w8",
]
