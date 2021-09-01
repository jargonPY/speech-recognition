import sys
import pathlib
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config
import speech_recognition.utils as utils
import json

"""
  - Try different preprocessing during training 
    (several epochs one way and several epochs another way)
  - Ensembles

  1. need to log / visualize the preprocessed data, make sure its properly processed
  2. need to create a local pipeline for testing/debugging 

  MODELS = {
    model_name: {
      version: 4,
      best_score: 0.8,
      best_version: 2
      ...
    }
  }
"""

class ModelsMetadata():
  pass

  def open_file():
    with open("models_metadata.json") as f:
      models_metadata = json.load(f)

  def update_file(data):
    with open("models_metadata.json", "w") as f:
      json.dump(data, f)

class BaseModel():

  def __init__(self, model_name, load_model, version):
    """
      - store path
      If dir does not exist:
        - create model dir
        - instantiate version (add to config JSON)
        - create version dir
      If dir does exist:
        - if version number specified in arg load version else load latest
        - if new params are specified create new version (should be decided within class or within main?) --> class

      version and params should not both be specified
    """

    self.version = 0
    self.model_path = str(pathlib.Path(__file__).parent) + "/" + model_name

    if not os.path.exists(str(pathlib.Path(__file__).parent) + "/models_metadata.json"):
      pathlib.Path("models_metadata.json").touch()
    with open("models_metadata.json") as f:
      self.models_metadata = json.load(f)

    if load_model:
      if version:
        self.version = version
      else:
        self.version = config.MODELS[model_name].latest_version

      # make sure a version exists
      self.version_path = self.model_path + f"/version_{self.version}"
      assert os.path.exists(self.version_path)
    
    else:
      if not os.path.exists(self.model_path):
        os.mkdir(self.model_path)
        config.MODELS[model_name] = {"latest_version": self.version}

      else:
        self.version = config.MODELS[model_name].latest_version + 1
        config.MODELS[model_name].latest_version += 1

      self.version_path = self.model_path + f"/version_{self.version}"
      os.mkdir(self.version_path)
      
  def update_version(self):
    pass

  def load_weights(self):

    if os.path.exists(self.version_path):
      #self.model.load_weights(weight_file_path)
      pass

  def fit(self, split=0.9, batch_size=128, epochs=10):
    """
      Save all weights, when resuming training or doing predictions
      load the latest.
        - how to find latest
        - eventually during predictions we'd want to load the best result (maybe save best only)
        - need to plot progress over several training runs 
            (probably needs custom callbacks for that since we are not saving the model/history)
    """

    train_set, val_set = utils.DataGenerator.get_file_names(split=split)
    train_generator = utils.DataGenerator(train_set, batch_size, one_hot=True)
    val_generator = utils.DataGenerator(val_set, batch_size, one_hot=True)

    checkpoint = keras.callbacks.ModelCheckpoint(
      self.version_path + "/{epoch:02d}-{val_loss:.2f}",
      save_weights_only=True
    )

    history = self.model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=epochs,
                        #max_queue_size=1,
                        workers=6,
                        use_multiprocessing=True,
                        callbacks=[checkpoint])

    #self.model.save_weights(weight_file_path)
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


class VanillaLSTM(BaseModel):

  MODEL_NAME = "vanilla_lstm"

  def __init__(self, load_model, version, audio_dim=26, hidden_dim=32):

    super().__init__(VanillaLSTM.MODEL_NAME, load_model, version)

    # training encoder
    encoder_inputs = Input(shape=(None, audio_dim), name="audio-input")
    encoder = LSTM(hidden_dim, return_state=True, name="audio-encoder")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # training decoder
    decoder_inputs = Input(shape=(None, config.NUM_CLASSES), name="text-input")
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, name="decoder")
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(config.NUM_CLASSES, activation="softmax", name="output")
    decoder_outputs = decoder_dense(decoder_outputs)

    # training model
    self.model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # inference encoder model
    self.encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

    # inference decoder model
    dec_state_h_input = Input((hidden_dim,), name = 'decoder_input_state_h')
    dec_state_c_input = Input((hidden_dim,), name = 'decoder_input_state_c')
    dec_states_input = [dec_state_h_input, dec_state_c_input]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=dec_states_input)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    self.decoder_model = tf.keras.Model([decoder_inputs, dec_states_input], [decoder_outputs, decoder_states])
    print("DONE")
    if load_model:
      self.load_weights()