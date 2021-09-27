import tensorflow as tf
import sys
import pathlib
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from models.model_metadata import ModelsMetadata

class LogsCallback(tf.keras.callbacks.Callback, ModelsMetadata):

    def __init__(self, version, model_name):
        ModelsMetadata.__init__(self)
        self.version = version
        self.model_name = model_name

    def on_train_begin(self):
        if not self.version in self.models_metadata[self.model_name]:
            self.models_metadata[self.model_name][self.version] = {} # create a helper function
        if not "train" in self.models_metadata[self.model_name][self.version]:
            self.models_metadata[self.model_name][self.version] = {}
        if not "loss" in self.models_metadata[self.model_name][self.version]:
            self.models_metadata[self.model_name][self.version]["loss"] = []
        if not "accuracy" in self.models_metadata[self.model_name][self.version]:
            self.models_metadata[self.model_name][self.version]["accuracy"] = []

    def on_train_batch_end(self, batch, logs=None):
        self.append_training_metric(self.model_name, self.model_version, "loss", logs["loss"])
        self.append_training_metric(self.model_name, self.model_version, "accuracy", logs["accuracy"])

