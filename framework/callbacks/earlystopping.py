from ..callback import Callback
from ..learner import CancelTrainException
from ..learner import Learner
import torch
from pathlib import Path

class EarlyStoppingCallback(Callback):

    def __init__(self, path:Path, epochs_to_wait=10, ignore_epochs=10, **kwargs):
        super(EarlyStoppingCallback, self).__init__(**kwargs)

        self.epochs_to_wait = epochs_to_wait
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.ignore_epochs = ignore_epochs

        self.path = path

    def after_validation(self, learner: Learner):
        if learner.loss.mean() < self.best_loss:
            torch.save(learner.model, self.path)
            self.best_epoch = learner.epoch
            self.best_loss = learner.loss.mean()

        if learner.epoch > self.ignore_epochs and learner.epoch - self.best_epoch > self.epochs_to_wait:
            raise CancelTrainException()

            