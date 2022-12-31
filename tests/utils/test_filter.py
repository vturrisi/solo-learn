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

import math

import torch
import torch.nn as nn
from solo.utils.misc import FilterInfNNan, filter_inf_n_nan


def test_filter():
    tensor = torch.randn(100)
    tensor[10] = math.nan
    filtered, selected = filter_inf_n_nan(tensor, return_indexes=True)
    assert filtered.size(0) == 99
    assert selected.sum() == 99

    tensor2 = torch.randn(100)
    assert [t.size(0) == 99 for t in filter_inf_n_nan([tensor, tensor2])]

    tensor = torch.randn(100, 30)
    tensor[10, 0] = math.inf
    assert filter_inf_n_nan(tensor).size(0) == 99

    tensor2 = torch.randn(100, 30)
    assert [t.size(0) == 99 for t in filter_inf_n_nan([tensor, tensor2])]

    class DummyNanLayer(nn.Module):
        def forward(self, x):
            x[10] = -math.inf
            return x

    dummy_nan_layer = DummyNanLayer()
    filter_layer = FilterInfNNan(dummy_nan_layer)
    assert filter_layer(tensor).size(0) == 99
