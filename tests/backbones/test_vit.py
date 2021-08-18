import torch
from solo.backbones import vit_tiny, vit_small, vit_base


def test_vit():
    dummy_data = torch.randn(10, 3, 224, 224)

    model = vit_tiny()
    out = model(dummy_data)
    assert out.size() == (10, 192)

    model = vit_small()
    out = model(dummy_data)
    assert out.size() == (10, 384)

    model = vit_base()
    out = model(dummy_data)
    assert out.size() == (10, 768)
