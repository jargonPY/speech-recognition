import string
import sys
import pathlib
import numpy as np
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config

class PreprocessTextLayer(tf.keras.layers.Layer):
  """
    Goal:
      Takes raw text files and produces preprocessed data ready to be trained on.
      By creating a preprocessing layer we can make our model truly "end-to-end"
      model, passing in raw data and producing predictions.

    Improvements:
      - clean_text, pad, one-hot encode need to be done sequentially
        but they can each (the for loop) be performed in parallel

      - @tf.function
          - https://www.tensorflow.org/guide/function
          - dont rely on Python side effects like object mutation or list appends
  """

  def __init__(self, one_hot_encode=True, **kwargs):
    super().__init__(**kwargs)
    self.one_hot_encode = one_hot_encode

  def clean_text(self, text_files):
 
    for text_file in text_files:
      text_file = text_file.translate(str.maketrans('', '', string.punctuation))
      text_file = text_file.translate(str.maketrans('', '', string.digits))
      text_file = text_file.lower().strip()

      text_file = ['<space>' if char == ' ' else char for char in text_file]
      text_file.insert(0, '<sos>')
      text_file.append('<eos>')
    return text_files

  def pad_text(self, text):
    return tf.keras.preprocessing.sequence.pad_sequences(text, padding="post", value="<pad>")

  def one_hot_encoder(self, text):
    """
      Custom one-hot encoder ensures that padding is not encoded as a seperate class but rather as
      a vector of zeros which can be masked in later layers.
    """

    one_hot = np.zeros(shape=(text.shape[0], text.shape[1], config.NUM_CLASSES))

    for sample_index, sample in enumerate(text):
      sample_encoding = np.zeros(shape=(text.shape[1], config.NUM_CLASSES))

      for token_index, token in enumerate(sample):
        if token == "<pad>": continue
        sample_encoding[token_index, config.TOKEN_TO_INDEX[token]] = 1.0

      one_hot[sample_index] = sample_encoding

    return one_hot[:, :-1, :], one_hot[:, 1:, :]

  def tokenizer(self, text):
    pass

  def call(self, text_files):
    """
      text_files: 2D array (files, text)
    """
    text = self.clean_text(text_files)
    text = self.pad_text(text)

    if self.one_hot_encode:
      decoder_input, decoder_output = self.one_hot_encoder(text)
    else:
      decoder_input, decoder_output = self.tokenizer(text)

    return decoder_input, decoder_output
    
