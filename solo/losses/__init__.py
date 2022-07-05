# Copyright 2022 solo-learn development team.

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

from solo.losses.barlow import barlow_loss_func
from solo.losses.byol import byol_loss_func
from solo.losses.deepclusterv2 import deepclusterv2_loss_func
from solo.losses.dino import DINOLoss
from solo.losses.mae import mae_loss_func
from solo.losses.mocov2plus import mocov2plus_loss_func
from solo.losses.mocov3 import mocov3_loss_func
from solo.losses.nnclr import nnclr_loss_func
from solo.losses.ressl import ressl_loss_func
from solo.losses.simclr import simclr_loss_func
from solo.losses.simsiam import simsiam_loss_func
from solo.losses.swav import swav_loss_func
from solo.losses.vibcreg import vibcreg_loss_func
from solo.losses.vicreg import vicreg_loss_func
from solo.losses.wmse import wmse_loss_func

__all__ = [
    "barlow_loss_func",
    "byol_loss_func",
    "deepclusterv2_loss_func",
    "DINOLoss",
    "mae_loss_func",
    "mocov2plus_loss_func",
    "mocov3_loss_func",
    "nnclr_loss_func",
    "ressl_loss_func",
    "simclr_loss_func",
    "simsiam_loss_func",
    "swav_loss_func",
    "vibcreg_loss_func",
    "vicreg_loss_func",
    "wmse_loss_func",
]
