import sys
import pathlib
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional
sys.path.append(str(pathlib.Path(__file__).parents[1]))
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import config
from models.base_model import BaseModel

class BidirectionalLSTM(BaseModel):

  MODEL_NAME = "bidirectional_lstm"

  def __init__(self, load_version, audio_dim=26, hidden_dim=32):

    super().__init__(BidirectionalLSTM.MODEL_NAME, load_version)

    encoder_inputs = Input(shape=(None, audio_dim), name="audio_input")
    decoder_inputs = Input(shape=(None, config.NUM_CLASSES), name="text_input")
    sequence_lengths = Input(shape=[])

    encoder = Bidirectional(LSTM(hidden_dim, return_state=True, name="audio_encoder"))
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_cell = tf.keras.layers.LSTMCell(512)
    output_layer = Dense(config.NUM_CLASSES)

    sampler = tfa.seq2seq.sampler.TrainingSampler()
    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                      output_layer=output_layer)

    final_outputs, final_state, final_sequence_lengths = decoder(
         decoder_inputs, initial_state=encoder_states,
         sequence_length=sequence_lengths)
    Y_proba = tf.nn.softmax(final_outputs.rnn_output)

    self.model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
                         outputs=[Y_proba])

    if load_version:
      self.load_weights()