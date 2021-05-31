import os
import json
from pytorch_lightning.callbacks import Callback


class EpochCheckpointer(Callback):
    def __init__(self, args, logdir="trained_models", frequency=25):
        self.args = args
        self.frequency = frequency
        self.logdir = logdir

    def initial_setup(self, trainer):
        if trainer.logger is None:
            version = None
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = os.path.join(self.logdir, version)
            self.ckpt_placeholder = f"{self.args.name}-{version}" + "-ep={}.ckpt"
        else:
            self.path = self.logdir
            self.ckpt_placeholder = f"{self.args.name}" + "-ep={}.ckpt"

        # create logging dirs
        if trainer.is_global_zero:
            try:
                os.makedirs(self.path)
            except:
                pass

    def save_args(self, trainer):
        if trainer.is_global_zero:
            args = vars(self.args)
            json_path = os.path.join(self.path, "args.json")
            json.dump(args, open(json_path, "w"))

    def save(self, trainer):
        epoch = trainer.current_epoch
        ckpt = self.ckpt_placeholder.format(epoch)
        trainer.save_checkpoint(os.path.join(self.path, ckpt))

    def on_train_start(self, trainer, pl_module):
        self.initial_setup(trainer)
        self.save_args(trainer)

    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.frequency == 0 and epoch != 0:
            self.save(trainer)

    def on_train_end(self, trainer, pl_module):
        self.save(trainer)
