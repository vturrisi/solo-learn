import json
import os
from argparse import ArgumentParser, Namespace
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

Checkpoint = namedtuple("Checkpoint", ["creation_time", "args", "checkpoint"])


class AutoResumer:
    SHOULD_MATCH = [
        "batch_size",
        "weight_decay",
        "lr",
        "dataset",
        "backbone",
        "max_epochs",
        "method",
        "name",
        "project",
        "entity",
    ]

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = Path("trained_models"),
        max_hours: int = 36,
    ):
        """Autoresumer object that automatically tries to find a checkpoint
        that is as old as max_time.

        Args:
            checkpoint_dir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to "trained_models".
            max_hours (int): maximum elapsed hours to consider checkpoint as valid.
        """

        self.checkpoint_dir = checkpoint_dir
        self.max_hours = timedelta(hours=max_hours)

    @staticmethod
    def add_autoresumer_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("autoresumer")
        parser.add_argument("--auto_resumer_max_hours", default=36, type=int)
        return parent_parser

    def find_checkpoint(self, args: Namespace):
        """Finds a valid checkpoint that matches the arguments

        Args:
            args (Namespace): namespace object containing all settings of the model.
        """

        current_time = datetime.now()

        possible_checkpoints = []
        for rootdir, _, files in os.walk(self.checkpoint_dir):
            rootdir = Path(rootdir)
            if files:
                # skip checkpoints that are empty
                try:
                    checkpoint_file = [rootdir / f for f in files if f.endswith(".ckpt")][0]
                except:
                    continue

                creation_time = datetime.fromtimestamp(os.path.getctime(checkpoint_file))
                if current_time - creation_time < self.max_hours:
                    ck = Checkpoint(
                        creation_time=creation_time,
                        args=rootdir / "args.json",
                        checkpoint=checkpoint_file,
                    )
                    possible_checkpoints.append(ck)

        if possible_checkpoints:
            # sort by most recent
            possible_checkpoints = sorted(
                possible_checkpoints, key=lambda ck: ck.creation_time, reverse=True
            )

            for checkpoint in possible_checkpoints:
                checkpoint_args = Namespace(**json.load(open(checkpoint.args)))
                if all(
                    getattr(checkpoint_args, param) == getattr(args, param)
                    for param in AutoResumer.SHOULD_MATCH
                ):
                    return checkpoint.checkpoint

        return None
