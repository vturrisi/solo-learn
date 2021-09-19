import torch
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)


def test_backbones():
    # swin models
    dummy_data = torch.randn(6, 3, 32, 32)
    model = swin_tiny(window_size=4, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = swin_small()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = swin_base()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = swin_large()
    assert isinstance(model(dummy_data), torch.Tensor)

    # vit models
    dummy_data = torch.randn(6, 3, 32, 32)
    model = vit_tiny(patch_size=8, img_size=32)
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = vit_small()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = vit_base()
    assert isinstance(model(dummy_data), torch.Tensor)

    dummy_data = torch.randn(6, 3, 224, 224)
    model = vit_large()
    assert isinstance(model(dummy_data), torch.Tensor)
