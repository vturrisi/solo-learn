from pytorch_lightning.callbacks import Callback


class StaticLR(Callback):
    def __init__(self, parameters):
        self.parameters = parameters

    def on_train_batch_start(self, trainer):
        print(dir(trainer))
        exit()
