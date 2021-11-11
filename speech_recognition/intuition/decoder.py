import tensorflow as tf

class Decoder(tf.keras.layers.Layer):

  def __init__(self, cell, sampler, output_layer, **kwargs):
    super().__init__(**kwargs)
    self.cell = cell
    self.sampler = sampler
    self.output_layer = output_layer

  def build(self, input_shape):
    pass

  def call(self, x, initial_state=None, training=None):
    pass

  def step(self, time, inputs, state, training=None):
    pass

  """
    encoded_state = Encoder()(inputs)
    output = Decoder()(encoded_state)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[output])
  """