import keras
import keras.callbacks

from holo.__typing import Literal, get_args
from holo.profilers import StopWatch




# on_train_begin 
#   on_epoch_begin
#       -> started training
#       -> finished testing
#       on_test_begin
#          -> started testing
#          -> finished testing
#       on_test_end
#   on_epoch_end
#   ... for each epoch
# on_train_end






class MesureFittingTimes(keras.callbacks.Callback):
    
    categories = Literal["train", "train_batch", "test", "test_batch"]
    categoriesValues: "list[categories]" = list(get_args(categories))
    
    def __init__(self):
        super().__init__()
        self.profilers: "dict[MesureFittingTimes.categories, StopWatch]" = \
            {c: StopWatch() for c in MesureFittingTimes.categoriesValues}
        self.epochs: "list[int]" = []
        """the epochs (index) done"""
        self.values: "dict[MesureFittingTimes.categories, dict[int, float]]" = \
            {c: {} for c in MesureFittingTimes.categoriesValues}
    
    def resetProfilers(self)->None:
        for prof in self.profilers.values():
            prof.reset()
    
    
    def on_epoch_begin(self, epoch:int, logs=None):
        # prepare the mesures
        self.resetProfilers()
        self.epochs.append(epoch)
        # start training
        self.profilers["train"].start()
        self.profilers["train_batch"].start(paused=True)
        
    def on_epoch_end(self, epoch:int, logs=None):
        # ensure all stoped
        running = {categorie for categorie, prof in self.profilers.items()
                    if prof.stoped is False}
        if len(running) != 0:
            raise RuntimeError(f"the following profs aren't stoped: {running}")
        # add the mesures
        for name in ("train", "test"):
            self.values[name][epoch] = self.profilers[name].mesuredTime
        for name in ("train_batch", "test_batch"):
            self.values[name][epoch] = \
                (self.profilers[name].mesuredTime / self.profilers[name].nbMesuresFinished)
    
    def on_test_begin(self, logs=None):
        # stop training
        self.profilers["train"].stop()
        self.profilers["train_batch"].stop()
        # start testing
        self.profilers["test"].start()
        self.profilers["test_batch"].start(paused=True)
    def on_test_end(self, logs=None):
        # stop testing
        self.profilers["test"].stop()
        self.profilers["test_batch"].stop()
    
    
    def on_train_batch_begin(self, batch:int, logs=None):
        self.profilers["train_batch"].play()
    def on_train_batch_end(self, batch:int, logs=None):
        self.profilers["train_batch"].pause()
    
    def on_test_batch_begin(self, batch:int, logs=None):
        self.profilers["test_batch"].play()
    def on_test_batch_end(self, batch:int, logs=None):
        self.profilers["test_batch"].pause()