from solo.losses.barlow import barlow_loss_func
from solo.losses.byol import byol_loss_func
from solo.losses.moco import moco_loss_func
from solo.losses.nnclr import nnclr_loss_func
from solo.losses.simclr import simclr_loss_func
from solo.losses.simsiam import simsiam_loss_func
from solo.losses.swav import swav_loss_func
from solo.losses.vicreg import vicreg_loss_func

__all__ = [
    "barlow_loss_func",
    "byol_loss_func",
    "moco_loss_func",
    "nnclr_loss_func",
    "simclr_loss_func",
    "simsiam_loss_func",
    "swav_loss_func",
    "vicreg_loss_func",
]
