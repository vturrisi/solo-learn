import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class Checkpointer(Callback):
    def __init__(
        self,
        args: Namespace,
        logdir: Union[str, Path] = Path("trained_models"),
        frequency: int = 1,
        keep_previous_checkpoints: bool = False,
    ):
        """Custom checkpointer callback that stores checkpoints in an easier to access way.

        Args:
            args (Namespace): namespace object containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            frequency (int, optional): number of epochs between each checkpoint. Defaults to 1.
            keep_previous_checkpoints (bool, optional): whether to keep previous checkpoints or not.
                Defaults to False.
        """

        super().__init__()

        self.args = args
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.keep_previous_checkpoints = keep_previous_checkpoints

    @staticmethod
    def add_checkpointer_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("checkpointer")
        parser.add_argument("--checkpoint_dir", default=Path("trained_models"), type=Path)
        parser.add_argument("--checkpoint_frequency", default=1, type=int)
        return parent_parser

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            version = None
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = self.logdir / version
            self.ckpt_placeholder = f"{self.args.name}-{version}" + "-ep={}.ckpt"
        else:
            self.path = self.logdir
            self.ckpt_placeholder = f"{self.args.name}" + "-ep={}.ckpt"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def save_args(self, trainer: pl.Trainer):
        """Stores arguments into a json file.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero:
            args = vars(self.args)
            json_path = self.path / "args.json"
            json.dump(args, open(json_path, "w"), default=lambda o: "<not serializable>")

    def save(self, trainer: pl.Trainer):
        """Saves current checkpoint.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.is_global_zero and not trainer.running_sanity_check:
            epoch = trainer.current_epoch  # type: ignore
            ckpt = self.path / self.ckpt_placeholder.format(epoch)
            trainer.save_checkpoint(ckpt)

            if self.last_ckpt and self.last_ckpt != ckpt and not self.keep_previous_checkpoints:
                os.remove(self.last_ckpt)
            self.last_ckpt = ckpt

    def on_train_start(self, trainer: pl.Trainer, _):
        """Executes initial setup and saves arguments.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_validation_end(self, trainer: pl.Trainer, _):
        """Tries to save current checkpoint at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0:
            self.save(trainer)
