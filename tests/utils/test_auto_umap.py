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

import shutil

from solo.methods import BarlowTwins
from solo.utils.auto_umap import AutoUMAP

from ..methods.utils import gen_base_cfg, gen_trainer, prepare_dummy_dataloaders


def test_auto_umap():
    method_kwargs = {
        "proj_hidden_dim": 2048,
        "proj_output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.025,
    }
    cfg = gen_base_cfg("barlow_twins", batch_size=2, num_classes=100)
    cfg.method_kwargs = method_kwargs
    model = BarlowTwins(cfg)

    # UMAP
    cfg = AutoUMAP.add_and_assert_specific_cfg(cfg)
    auto_umap = AutoUMAP(cfg.name)

    trainer = gen_trainer(cfg, auto_umap)

    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        num_large_crops=cfg.data.num_large_crops,
        num_small_crops=cfg.data.num_small_crops,
        num_classes=cfg.data.num_classes,
        batch_size=cfg.optimizer.batch_size,
    )
    trainer.fit(model, train_dl, val_dl)

    # check if checkpointer dumped the umap
    umap_path = auto_umap.path / auto_umap.umap_placeholder.format(trainer.current_epoch - 1)
    assert umap_path.exists()

    # clean stuff
    shutil.rmtree(auto_umap.logdir)
