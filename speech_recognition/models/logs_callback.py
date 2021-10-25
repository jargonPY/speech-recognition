import sys
import pathlib
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[1]))
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import config
import utils

logger = utils.generate_logger(__name__, "main.log")
class LogsCallback(tf.keras.callbacks.Callback):

    def __init__(self, model_name, version):
        self.version = version
        self.model_name = model_name

    def on_train_begin(self, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        logger.info(f"batch: {batch}, logs: {logs}")
        config.document.append_metric(self.model_name, self.version, "train", "loss", logs["loss"])
        config.document.append_metric(self.model_name, self.version, "train", "accuracy", logs["accuracy"])

