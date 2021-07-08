from solo.methods.barlow_twins import BarlowTwins
from solo.methods.base import BaseModel
from solo.methods.byol import BYOL
from solo.methods.dino import DINO
from solo.methods.linear import LinearModel
from solo.methods.mocov2plus import MoCoV2Plus
from solo.methods.nnclr import NNCLR
from solo.methods.simclr import SimCLR
from solo.methods.simsiam import SimSiam
from solo.methods.swav import SwAV
from solo.methods.vicreg import VICReg
from solo.methods.wmse import WMSE

METHODS = {
    # base classes
    "base": BaseModel,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "nnclr": NNCLR,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "swav": SwAV,
    "vicreg": VICReg,
    "wmse": WMSE,
}


try:
    from solo.methods import dali
except ImportError:
    __all__ = [
        "BarlowTwins",
        "BYOL",
        "BaseModel",
        "DINO",
        "LinearModel",
        "MoCoV2Plus",
        "NNCLR",
        "SimCLR",
        "SimSiam",
        "SwAV",
        "VICReg",
        "WMSE",
    ]
else:
    __all__ = [
        "BarlowTwins",
        "BYOL",
        "BaseModel",
        "DINO",
        "LinearModel",
        "MoCoV2Plus",
        "NNCLR",
        "SimCLR",
        "SimSiam",
        "SwAV",
        "VICReg",
        "WMSE",
        "dali",
    ]
