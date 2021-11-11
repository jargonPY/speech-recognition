import numpy as np
import os
import tensorflow as tf
import sys
import pathlib
import random
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config
from speech_recognition.preprocessing import PreprocessAudio, PreprocessText, preprocess_text
from speech_recognition.utils.generate_logger import generate_logger

"""
Sequence are a safer way to do multiprocessing. This structure guarantees that the network will only train 
once on each sample per epoch which is not the case with generators.

https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
"""

logger = generate_logger(__name__, "main.log")

class DataGenerator(tf.keras.utils.Sequence):

  def __init__(self, samples, batch_size, one_hot=True):
    
    self.samples = samples
    self.batch_size = batch_size
    self.one_hot = one_hot

  def open_preprocessed_files(self, index):
    # load_preprocessed_files
    audio_data = []
    input_text_data = []
    output_text_data = []

    logger.info(f"Getting samples from: {index * self.batch_size} to: {(index + 1) * self.batch_size}")

    for sample in self.samples[index * self.batch_size:(index + 1) * self.batch_size]:
      audio_data.append(np.load(sample[0]))
      input_text_data.append(np.load(sample[1]))
      output_text_data.append(np.load(sample[2]))
      
    audio_data = tf.keras.preprocessing.sequence.pad_sequences(audio_data, padding='post')
    output_text_data = tf.keras.utils.to_categorical(output_text_data, config.NUM_CLASSES)

    if self.one_hot:
      input_text_data = tf.keras.utils.to_categorical(input_text_data, config.NUM_CLASSES)
    else:
      input_text_data = np.asarray(input_text_data)

    return audio_data, input_text_data, output_text_data

  def preprocess_files(self, index):
    # preprocess_real_time

    logger.info(f"Index: {index}. Getting samples from: {index * self.batch_size} to: {(index + 1) * self.batch_size}")

    audio_data = []
    input_text_data = []
    output_text_data = []

    preprocess_audio = PreprocessAudio()
    preprocess_text = PreprocessText()
    for sample in self.samples[index * self.batch_size:(index + 1) * self.batch_size]:
      audio = preprocess_audio.preprocess_file(sample[0])
      input_text, output_text = preprocess_text.preprocess_text(sample[1])
      audio_data.append(audio)
      input_text_data.append(input_text)
      output_text_data.append(output_text)

    # zero padding
    audio_data = tf.keras.preprocessing.sequence.pad_sequences(audio_data, padding='post')
    input_text_data = tf.keras.preprocessing.sequence.pad_sequences(input_text_data, padding='post')
    output_text_data = tf.keras.preprocessing.sequence.pad_sequences(output_text_data, padding='post')
    # one-hot encode output (necessary for categorical cross-entropy)
    output_text_data = tf.keras.utils.to_categorical(output_text_data, config.NUM_CLASSES)
    # one-hot encode input (necessary if there is no embedding layer)
    if self.one_hot:
      input_text_data = tf.keras.utils.to_categorical(input_text_data, config.NUM_CLASSES)
    
    return audio_data, input_text_data, output_text_data
  
  def on_epoch_end(self):
    """
      shuffle the dataset at the end of each epoch
    """
    random.shuffle(self.samples)

  def __len__(self):
    """
      returns the number of batches in the sequence 
        - return value must be of type int
    """
    return len(self.samples) // self.batch_size

  def __getitem__(self, index):
    """
      index - passed in by training model

      returns batch, (input, output) value pair, at position index
        - returned values must be numpy arrays
    """
    audio_data, input_text_data, output_text_data = self.preprocess_files(index)
    logger.info(f"Shapes: {audio_data.shape, input_text_data.shape, output_text_data.shape}")
    return (audio_data, input_text_data), output_text_data

  @staticmethod  
  def get_file_names(split=0.8):

    audio_files = sorted(os.listdir(config.AUTIO_TRAIN_PATH))
    text_files = sorted(os.listdir(config.TEXT_TRAIN_PATH))
    assert len(audio_files) == len(text_files), f"Number of files must be equal: Audio {len(audio_files)}, Text {len(text_files)}"

    file_names = []
    for index in range(len(audio_files)):
      audio_file = config.AUTIO_TRAIN_PATH + "/" + audio_files[index]
      text_file = config.TEXT_TRAIN_PATH + "/" + text_files[index]
      files = [audio_file, text_file]
      file_names.append(files)

    split_index = int(np.floor(len(file_names) * split))
    train_set = file_names[:split_index]
    val_set = file_names[split_index:]

    logger.info(f"Train set length: {len(train_set)}, Val set length: {len(val_set)}")
    return train_set, val_set

  @staticmethod
  def get_test_files():

    audio_files = sorted(os.listdir(config.AUTIO_TRAIN_PATH))
    text_files = sorted(os.listdir(config.TEXT_TRAIN_PATH))
    assert len(audio_files) == len(text_files), f"Number of files must be equal: Audio {len(audio_files)}, Text {len(text_files)}"

    file_names = []
    for index in range(len(audio_files)):
      audio_file = config.AUDIO_TEST_PATH + "/" + audio_files[index]
      text_file = config.TEXT_TEST_PATH + "/" + text_files[index]
      files = [audio_file, text_file]
      file_names.append(files)
    return file_names

