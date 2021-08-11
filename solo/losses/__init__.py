from solo.losses.barlow import barlow_loss_func
from solo.losses.byol import byol_loss_func
from solo.losses.deepclusterv2 import deepclusterv2_loss_func
from solo.losses.dino import DINOLoss
from solo.losses.moco import moco_loss_func
from solo.losses.nnclr import nnclr_loss_func
from solo.losses.ressl import ressl_loss_func
from solo.losses.simclr import manual_simclr_loss_func, simclr_loss_func
from solo.losses.simsiam import simsiam_loss_func
from solo.losses.swav import swav_loss_func
from solo.losses.vicreg import vicreg_loss_func
from solo.losses.wmse import wmse_loss_func

__all__ = [
    "barlow_loss_func",
    "byol_loss_func",
    "deepclusterv2_loss_func",
    "DINOLoss",
    "moco_loss_func",
    "nnclr_loss_func",
    "ressl_loss_func",
    "simclr_loss_func",
    "manual_simclr_loss_func",
    "simsiam_loss_func",
    "swav_loss_func",
    "vicreg_loss_func",
    "wmse_loss_func",
]
