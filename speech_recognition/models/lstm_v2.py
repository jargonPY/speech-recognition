from models.base_model import BaseModel
import config
import sys
import pathlib
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
sys.path.append(str(pathlib.Path(__file__).parents[1]))
sys.path.append(str(pathlib.Path(__file__).parents[2]))


class VanillaLSTMV2(BaseModel):

    MODEL_NAME = "vanilla_lstm_v2"

    def __init__(self, mode, load_version, audio_dim=26, hidden_dim=32):

        super().__init__(VanillaLSTMV2.MODEL_NAME, load_version)
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim

        if mode == "train":
            if load_version:
                self.model = tf.keras.models.load_model(self.version_path)
            else:
                self.model = self.build_training_model()
        else:
            model = tf.keras.models.load_model(self.version_path)
            self.encoder_model, self.decoder_model = self.build_inference_model()
            weights = model.get_weights()
            self.encoder_model.set_weights(weights)
            self.decoder_model.set_weights(weights)

    def encoder(self):

        encoder_inputs = Input(
            shape=(None, self.audio_dim), name="audio-input")
        encoder = LSTM(self.hidden_dim, return_state=True,
                       name="audio-encoder")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        return encoder_inputs, encoder_states

    def decoder(self, initial_state):

        decoder_inputs = Input(
            shape=(None, config.NUM_CLASSES), name="text-input")
        decoder_lstm = LSTM(self.hidden_dim, return_sequences=True,
                            return_state=True, name="decoder")
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=initial_state)
        decoder_states = [state_h, state_c]
        decoder_dense = Dense(config.NUM_CLASSES,
                              activation="softmax", name="output")
        decoder_outputs = decoder_dense(decoder_outputs)
        return decoder_inputs, decoder_outputs, decoder_states

    def build_training_model(self):

        encoder_inputs, encoder_states = self.encoder()

        decoder_inputs, decoder_outputs, decoder_states = self.decoder(
            initial_state=encoder_states)

        return tf.keras.Model(
            [encoder_inputs, decoder_inputs], decoder_outputs)

    def build_inference_model(self):

        encoder_inputs, encoder_states = self.encoder()
        encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

        dec_state_h_input = Input(
            (self.hidden_dim,), name='decoder_input_state_h')
        dec_state_c_input = Input(
            (self.hidden_dim,), name='decoder_input_state_c')
        dec_states_input = [dec_state_h_input, dec_state_c_input]

        decoder_inputs, decoder_outputs, decoder_states = self.decoder(
            initial_state=dec_states_input)
        decoder_model = tf.keras.Model([decoder_inputs, dec_states_input], [
            decoder_outputs, decoder_states])
        return encoder_model, decoder_model
