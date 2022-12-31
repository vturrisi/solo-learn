# Copyright 2023 solo-learn development team.

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

import json
import shutil

from omegaconf import OmegaConf
from solo.methods import BarlowTwins
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer

from ..methods.utils import gen_base_cfg, gen_trainer, prepare_dummy_dataloaders


def test_checkpointer():
    method_kwargs = {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.025,
    }
    cfg = gen_base_cfg("barlow_twins", batch_size=2, num_classes=100)
    cfg.method_kwargs = method_kwargs
    cfg = Checkpointer.add_and_assert_specific_cfg(cfg)
    model = BarlowTwins(cfg)

    # checkpointer
    ckpt_callback = Checkpointer(cfg)

    trainer = gen_trainer(cfg, ckpt_callback)

    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=cfg.data.num_small_crops,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)

    # check if checkpointer dumped the args
    args_path = ckpt_callback.path / "args.json"
    assert args_path.exists()

    # check if the args are correct
    loaded_cfg = json.load(open(args_path))
    cfg_dict = OmegaConf.to_container(cfg)
    for k in cfg_dict:
        assert cfg_dict[k] == loaded_cfg[k]

    auto_resumer = AutoResumer(ckpt_callback.logdir, max_hours=1)
    assert auto_resumer.find_checkpoint(cfg) is not None

    # check arguments
    cfg = auto_resumer.add_and_assert_specific_cfg(cfg)
    assert not OmegaConf.is_missing(cfg, "auto_resume")
    assert not OmegaConf.is_missing(cfg, "auto_resume.enabled")
    assert not OmegaConf.is_missing(cfg, "auto_resume.max_hours")

    # clean stuff
    shutil.rmtree(ckpt_callback.logdir)
