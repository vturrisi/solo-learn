from .barlow_twins import BarlowTwins
from .byol import BYOL
from .mocov2plus import MoCoV2Plus
from .nnclr import NNCLR
from .simclr import SimCLR
from .simsiam import SimSiam
from .swav import SwAV
from .vicreg import VICReg


METHODS = {
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "mocov2plus": MoCoV2Plus,
    "nnclr": NNCLR,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "swav": SwAV,
    "vicreg": VICReg
}
