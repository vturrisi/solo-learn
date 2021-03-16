from pytorch_lightning.callbacks import Callback


class StaticLR(Callback):
    def __init__(self, lrs, param_group_indexes):
        self.lrs = lrs
        self.param_group_indexes = param_group_indexes

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        for lr, idx in zip(self.lrs, self.param_group_indexes):
            trainer.optimizers[0].param_groups[idx]["lr"] = lr
