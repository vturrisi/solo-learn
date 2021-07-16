from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import umap
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import Callback

from .gather_layer import gather


class AutoUMAP(Callback):
    def __init__(self, frequency: int = 1, color_pallete: str = "hls"):
        super().__init__()
        self.frequency = frequency
        self.color_pallete = color_pallete

    @staticmethod
    def add_checkpointer_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("auto_umap")
        parser.add_argument("--auto_umap_frequency", default=1, type=int)
        return parent_parser

    def on_train_start(self, trainer: pl.Trainer, _):
        """Checks wandb is available

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        assert isinstance(trainer.logger, pl.loggers.WandbLogger)

    def plot(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the module.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
            module (pl.LightningModule): current module object.
        """

        device = module.device
        data = []
        Y = []

        # set module to eval model and collect all feature representations
        module.eval()
        with torch.no_grad():
            for x, y in trainer.val_dataloaders[0]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = module(x)["logits"]

                logits = gather(logits)
                y = gather(y)

                data.append(logits.cpu())
                Y.append(y.cpu())
        module.train()

        if trainer.is_global_zero and len(data):
            data = torch.cat(data, dim=0).numpy()
            Y = torch.cat(Y, dim=0).numpy()

            data = umap.UMAP(n_components=2).fit_transform(data)

            # assing to dataframe
            df = pd.DataFrame()
            df["feat_1"] = data[:, 0]
            df["feat_2"] = data[:, 1]
            df["Y"] = Y

            plt.figure(figsize=(9, 9))
            ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="Y",
                palette=sns.color_palette(self.color_pallete, len(np.unique(Y))),
                data=df,
                legend="full",
                alpha=0.3,
            )
            ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
            ax.tick_params(left=False, right=False, bottom=False, top=False)
            plt.tight_layout()

            wandb.log(
                {"validation_umap": wandb.Image(ax)}, commit=False,
            )
            plt.close()

    def on_validation_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Tries to generate an up-to-date UMAP visualization of the features
        at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0 and not trainer.running_sanity_check:
            self.plot(trainer, module)
