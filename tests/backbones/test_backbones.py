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

import torch
from solo.backbones import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    resnet18,
    resnet50,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
    wide_resnet28w2,
    wide_resnet28w8,
)


def test_backbones():
    # swin models
    dummy_data = torch.randn(6, 3, 32, 32)
    model = swin_tiny(method=None, window_size=4, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = swin_small(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = swin_base(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = swin_large(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    # vit models
    dummy_data = torch.randn(6, 3, 32, 32)
    model = vit_tiny(method=None, patch_size=8, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_tiny(method="mocov3", patch_size=8, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_tiny(method="mae", patch_size=8, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = vit_small(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_base(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_large(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_small(method="mocov3")
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_base(method="mocov3")
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_large(method="mocov3")
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_small(method="mae")
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_base(method="mae")
    assert isinstance(model(dummy_data), torch.Tensor)

    model = vit_large(method="mae")
    assert isinstance(model(dummy_data), torch.Tensor)

    # PoolFormer
    dummy_data = torch.randn(6, 3, 32, 32)
    model = poolformer_s12(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = poolformer_s24(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = poolformer_s36(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = poolformer_m36(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = poolformer_m48(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    # ConvNeXt
    dummy_data = torch.randn(6, 3, 32, 32)
    model = convnext_tiny(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = convnext_small(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = convnext_base(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    model = convnext_large(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    # WideResnet
    dummy_data = torch.randn(6, 3, 32, 32)
    model = wide_resnet28w2(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = wide_resnet28w8(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    # Resnet
    dummy_data = torch.randn(6, 3, 32, 32)
    model = resnet18(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = resnet50(method=None)
    assert isinstance(model(dummy_data), torch.Tensor)
