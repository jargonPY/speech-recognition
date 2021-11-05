import sys
import pathlib
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
sys.path.append(str(pathlib.Path(__file__).parents[1]))
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import config
from models.base_model import BaseModel
class VanillaLSTM(BaseModel):

  MODEL_NAME = "vanilla_lstm"

  def __init__(self, load_version, audio_dim=26, hidden_dim=32):
    
    super().__init__(VanillaLSTM.MODEL_NAME, load_version)

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
    
    if load_version:
      self.load_weights()