from solo.utils import (
    checkpointer,
    classification_dataloader,
    gather_layer,
    lars,
    metrics,
    momentum,
    pretrain_dataloader,
    sinkhorn_knopp,
)

__all__ = [
    "classification_dataloader",
    "pretrain_dataloader",
    "checkpointer",
    "gather_layer",
    "lars",
    "metrics",
    "momentum",
    "sinkhorn_knopp",
]

try:
    from solo.utils import dali_dataloader  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali_dataloader")

try:
    from solo.utils import auto_umap  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("auto_umap")
