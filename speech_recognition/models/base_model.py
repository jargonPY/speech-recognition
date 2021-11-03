import sys
import pathlib
import os
import datetime
import numpy as np
from numpy.lib.recfunctions import _append_fields_dispatcher
import tensorflow as tf
from tensorboard import program
sys.path.append(str(pathlib.Path(__file__).parents[1]))
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import config
import utils
from models.logs_callback import LogsCallback
from speech_recognition.preprocessing import PreprocessAudio

logger = utils.generate_logger(__name__, "main.log")
class BaseModel():

  def __init__(self, model_name, load_version):
    """
      - store version path
      If dir does not exist:
        - create model dir
        - instantiate version (add to config JSON)
        - create version dir
      If dir does exist:
        - if version number specified in arg load version else load latest
    """

    self.model_name = model_name
    self.model_path = str(pathlib.Path(__file__).parent) + "/" + model_name
    self.version = None
    self.version_path = None

    if load_version:
      if load_version == "latest":
        self.version = config.document.get_latest_version(model_name)

      elif config.document.check_version_exists(model_name, load_version):
        self.version = load_version

      else:
        logger.error("Version specified does not exist")
        raise ValueError("The version does not exist")
      self.version_path = self.model_path + f"/version_{self.version}"

    else:
      if not os.path.exists(self.model_path):
        os.mkdir(self.model_path)
        config.document.add_model(model_name)
        self.version = 1

      else:
        self.version = config.document.update_version(model_name)

      self.version_path = self.model_path + f"/version_{self.version}"
      os.mkdir(self.version_path)

      logger.info(f"model: {self.model_name}-{self.model_path}, version: {self.version}-{self.version_path}")

  def load_weights(self):

    if os.path.exists(self.version_path):
      self.model.load_weights(self.version_path + "/checkpoint")
      # log some of the model weights/layers to ensure proper loading of each layer

  def fit(self, split=0.9, batch_size=128, epochs=2):
    """
      Save all weights, when resuming training or doing predictions
      load the latest.
        - how to find latest
        - eventually during predictions we'd want to load the best result (maybe save best only)
        - need to plot progress over several training runs 
            (probably needs custom callbacks for that since we are not saving the model/history)
    """

    train_set, val_set = utils.DataGenerator.get_file_names(split=split)

    train_set = train_set[:batch_size]
    val_set = val_set[:batch_size]

    train_generator = utils.DataGenerator(train_set, batch_size, one_hot=True)
    val_generator = utils.DataGenerator(val_set, batch_size, one_hot=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
      #self.version_path + "/{epoch:02d}",
      self.version_path + "/checkpoint", # when saving every version also change load_weights
      save_weights_only=True,
      save_freq="epoch"
    )

    #log_dir = f"{self.version_path}/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #log_dir = f"{self.version_path}/logs"
    # to open tensorboard: tensorboard --logdir={working_dir/.../version_path/logs/}
    #tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    logsCallback = LogsCallback(self.model_name, self.version)
    
    self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    history = self.model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        max_queue_size=1,
                        workers=1,
                        use_multiprocessing=True,
                        callbacks=[checkpoint, logsCallback])
    return history

  def evaluate(self):
    pass

  def test(self):
    # combine predict and evaluate
    test_set = utils.DataGenerator.get_test_files()
    self.predict(test_set[0][0]) #temp for debugging

  def predict(self, audio_input):

    audio_input = PreprocessAudio().preprocess_file(audio_input)
    audio_input = np.reshape(audio_input, (1, -1, audio_input.shape[1]))
    encoder_state = self.encoder_model.predict(audio_input)

    # (1 sample, 1 timestep, NUM_CLASSES possibilties) --> should include zero padding as a class
    target_seq = np.zeros((1, 1, config.NUM_CLASSES))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 1] = 1.0

    # Sampling loop for a batch of sequences (here we assume a batch size of 1)
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
      # internal_state = [hidden_state, cell_state]
      output_tokens, internal_state = self.decoder_model.predict([target_seq] + encoder_state)

      # Sample a token (output_tokens.shape = (1,1,30))
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_char = config.INDEX_TO_TOKEN[sampled_token_index]
      decoded_sentence += sampled_char

      # Exit condition: either hit max length or find stop character
      if sampled_char == "<eos>" or len(decoded_sentence) > config.MAX_DECODER_SEQ_LENGTH:
          stop_condition = True

      # Update the target sequence (of length 1).
      target_seq = np.zeros((1, 1, config.NUM_CLASSES))
      # sampled_token_index does not start at zero does that matter?
      target_seq[0, 0, sampled_token_index] = 1.0

      # Update states
      encoder_state = internal_state
    print("Decoded sentence: ", decoded_sentence)

