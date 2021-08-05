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
        "output_dim": 2048,
        "lamb": 5e-3,
        "scale_loss": 0.001,
    }

    # normal training
    BASE_KWARGS = gen_base_kwargs(cifar=False, multicrop=False)
    kwargs = {**BASE_KWARGS, **DATA_KWARGS, **method_kwargs}
    model = BarlowTwins(**kwargs)

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
        num_crops=BASE_KWARGS["num_crops"],
        num_small_crops=0,
        num_classes=BASE_KWARGS["num_classes"],
        multicrop=False,
    )

    trainer.fit(model, train_dl, val_dl)

    # check if checkpointer dumped the umap
    umap_path = auto_umap.path / auto_umap.umap_placeholder.format(trainer.current_epoch)
    assert umap_path.exists()

    # clean stuff
    shutil.rmtree(auto_umap.logdir)
