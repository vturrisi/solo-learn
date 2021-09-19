"""
Copy-pasted from timm, but allowing different window sizes.

"""


from timm.models.swin_transformer import _create_swin_transformer, register_model
from timm.models.vision_transformer import _create_vision_transformer


@register_model
def swin_tiny(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer("swin_tiny_patch4_window7_224", **model_kwargs)


@register_model
def swin_small(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer(
        "swin_small_patch4_window7_224", pretrained=False, **model_kwargs
    )


@register_model
def swin_base(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer(
        "swin_base_patch4_window7_224", pretrained=False, **model_kwargs
    )


@register_model
def swin_large(window_size=7, **kwargs):
    model_kwargs = dict(
        patch_size=4,
        window_size=window_size,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        num_classes=0,
        **kwargs,
    )
    return _create_swin_transformer(
        "swin_large_patch4_window7_224", pretrained=False, **model_kwargs
    )


@register_model
def vit_tiny(patch_size=16, **kwargs):
    """ViT-Tiny (Vit-Ti/16)"""
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, num_classes=0, **kwargs
    )
    model = _create_vision_transformer("vit_tiny_patch16_224", pretrained=False, **model_kwargs)
    return model


@register_model
def vit_small(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, num_classes=0, **kwargs
    )
    model = _create_vision_transformer("vit_small_patch16_224", pretrained=False, **model_kwargs)
    return model


@register_model
def vit_base(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, num_classes=0, **kwargs
    )
    model = _create_vision_transformer("vit_base_patch16_224", pretrained=False, **model_kwargs)
    return model


@register_model
def vit_large(patch_size=16, **kwargs):
    model_kwargs = dict(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, num_classes=0, **kwargs
    )
    model = _create_vision_transformer("vit_large_patch16_224", pretrained=False, **model_kwargs)
    return model
