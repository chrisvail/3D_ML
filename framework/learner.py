from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


class Learner:

    def __init__(self, model: nn.Module, optimiser, loss_func, dataloaders: dict[torch.utils.data.DataLoader], callbacks=None):
        self.model = model
        self.optimiser = optimiser
        self.loss_func = loss_func
        self.dataloaders = dataloaders
        self.callbacks = [] if callbacks is None else list(callbacks)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self, epochs=1, validation_epochs=1):
        self.num_epochs = epochs
        try: 
            with self._callback_manager("train"):
                for i in range(epochs):
                    self.epoch = i
                    if "train" in self.dataloaders:
                        try: 
                            with self._callback_manager("epoch"):
                                self.one_epoch()
                                if "validation" in self.dataloaders and self.epoch % validation_epochs == 0: 
                                    with self._callback_manager("validation"):
                                        self._validate()
                        except CancelEpochException:
                            pass

                    
        except CancelTrainException:
            pass

        if "test" in self.dataloaders:
            with self._callback_manager("test"):
                self._test()

    def one_epoch(self):
        self.model.train(True)
        for j, (data, labels) in enumerate(self.dataloaders["train"]):
            data = data.to(self.device).float()
            labels = labels.to(self.device)
            self.batch_num = j
            self.data = data
            self.labels = labels
            try:
                with self._callback_manager("batch"):
                    self.predictions = self.model(self.data)

                    self.loss = self.loss_func(self.predictions, self.labels)
                     
                    self.loss.mean().backward()

                    self.optimiser.step()

                    self.optimiser.zero_grad()
            except CancelBatchException:
                pass

    def _validate(self):
        self.model.train(False)
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.dataloaders["validation"]):
                data = data.to(self.device).float()
                labels = labels.to(self.device)
                self.data, self.labels = data, labels
                with self._callback_manager("validation_batch"):
                    self.predictions = self.model(self.data)
                    self.loss_batch = self.loss_func(self.predictions, self.labels)

                    if i:
                        self.loss = torch.cat((self.loss, self.loss_batch), dim=0)
                    else:
                        self.loss = self.loss_batch

    def _test(self):
        self.model.train(False)
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.dataloaders["test"]):
                data = data.to(self.device).float()
                labels = labels.to(self.device)
                self.data, self.labels = data, labels
                with self._callback_manager("test_batch"):
                    self.predictions = self.model(self.data)
                    self.loss_batch = self.loss_func(self.predictions, self.labels)

                    if i:
                        self.loss = torch.cat((self.loss, self.loss_batch), dim=0)
                    else:
                        self.loss = self.loss_batch
        

    @contextmanager
    def _callback_manager(self, name):
        # Is there a way to move the try except block into here?
        # Might have to pass in a list of exceptions in here but thats doable
        # That way I could run the after callbacks even if the context was cancelled early
        # Also would be nice to have a hook to let callbacks know if the batch/epoch/training was cancelled early
        # Something like self.epoch_failed = True but more computational than that
        self._run_callback("before_" + name)
        yield
        self._run_callback("after_" + name)


    def _run_callback(self, name):
        for callback in sorted(self.callbacks, key=lambda x: x.order):
            callback = getattr(callback, name, None)

            if callback is not None:
                callback(self)
