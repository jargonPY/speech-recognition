import sys
import pathlib
import os
import numpy as np
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config
import speech_recognition.utils as utils
from models.model_metadata import ModelsMetadata

class BaseModel(ModelsMetadata):

  def __init__(self, model_name, load_model, version):
    """
      - store version path
      If dir does not exist:
        - create model dir
        - instantiate version (add to config JSON)
        - create version dir
      If dir does exist:
        - if version number specified in arg load version else load latest
        - if new params are specified create new version (should be decided within class or within main?) --> class
    """

    super().__init__()
    self.model_path = str(pathlib.Path(__file__).parent) + "/" + model_name

    if load_model:
      if version and self.check_version_exists(model_name, version):
        self.version = version
      else:
        self.version = self.get_latest_model_version(model_name)
    
      self.version_path = self.model_path + f"/version_{self.version}"

    else:
      if not os.path.exists(self.model_path):
        os.mkdir(self.model_path)
        self.version = self.add_new_model(model_name)

      else:
        # if this is the first version of the model, make sure to include --load_model
        self.version = self.update_model_version(model_name)

      self.version_path = self.model_path + f"/version_{self.version}"
      os.mkdir(self.version_path)
      
  def update_version(self):
    pass

  def load_weights(self):

    if os.path.exists(self.version_path):
      self.model.load_weights(self.version_path + "/checkpoint")

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
      #self.version_path + "/{epoch:02d}-{val_loss:.2f}",
      self.version_path + "/checkpoint",
      save_weights_only=True,
      save_freq="epoch"
    )
    
    self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    history = self.model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        max_queue_size=1,
                        workers=1,
                        use_multiprocessing=True,
                        callbacks=[checkpoint])
    return history

  def test(self):
    pass

  def predict(self, audio_input):
    # Encode the input as state vectors.
    encoder_state = self.encoder_model.predict(audio_input)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, config.NUM_CLASSES))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, config.TOKEN_TO_INDEX] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, hidden_state, cell_state = self.decoder_model.predict([target_seq] + encoder_state)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = config.INDEX_TO_TOKEN[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "<eos>" or len(decoded_sentence) > config.MAX_DECODER_SEQ_LENGTH:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, config.NUM_CLASSES))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        encoder_state = [hidden_state, cell_state]

