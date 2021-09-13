from solo.methods.barlow_twins import BarlowTwins
from solo.methods.base import BaseMethod
from solo.methods.byol import BYOL
from solo.methods.deepclusterv2 import DeepClusterV2
from solo.methods.dino import DINO
from solo.methods.linear import LinearModel
from solo.methods.mocov2plus import MoCoV2Plus
from solo.methods.nnclr import NNCLR
from solo.methods.ressl import ReSSL
from solo.methods.simclr import SimCLR
from solo.methods.simsiam import SimSiam
from solo.methods.swav import SwAV
from solo.methods.vicreg import VICReg
from solo.methods.wmse import WMSE

METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "nnclr": NNCLR,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "swav": SwAV,
    "vicreg": VICReg,
    "wmse": WMSE,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "MoCoV2Plus",
    "NNCLR",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SwAV",
    "VICReg",
    "WMSE",
]

try:
    from solo.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
