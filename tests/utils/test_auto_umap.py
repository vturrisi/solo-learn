# Copyright 2021 solo-learn development team.

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

import argparse
import shutil

from pytorch_lightning import Trainer
from solo.methods import BarlowTwins
from solo.utils.auto_umap import AutoUMAP

from ..methods.utils import DATA_KWARGS, gen_base_kwargs, prepare_dummy_dataloaders


def test_auto_umap():
    method_kwargs = {
        "name": "barlow_twins",
        "proj_hidden_dim": 2048,
        "proj_output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.001,
    }

    # normal training
    BASE_KWARGS = gen_base_kwargs(cifar=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = BarlowTwins(**kwargs, disable_knn_eval=True)

    args = argparse.Namespace(**kwargs)

    # UMAP
    auto_umap = AutoUMAP(args)

    trainer = Trainer.from_argparse_args(
        args,
        checkpoint_callback=False,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[auto_umap],
    )

    train_dl, val_dl = prepare_dummy_dataloaders(
        "imagenet100",
        num_large_crops=BASE_KWARGS["num_large_crops"],
        num_small_crops=0,
        num_classes=BASE_KWARGS["num_classes"],
        multicrop=False,
        batch_size=BASE_KWARGS["batch_size"],
    )
    model.set_loaders(train_loader=train_dl, val_loader=val_dl)
    trainer.fit(model, train_dl, val_dl)

    # check if checkpointer dumped the umap
    umap_path = auto_umap.path / auto_umap.umap_placeholder.format(trainer.current_epoch)
    assert umap_path.exists()

    # clean stuff
    shutil.rmtree(auto_umap.logdir)
