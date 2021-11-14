from speech_recognition.utils.load_audio import load_audio
from speech_recognition.utils.generate_logger import generate_logger
from speech_recognition.preprocessing import PreprocessAudioLayer, PreprocessTextLayer
import speech_recognition.config as config
import numpy as np
import os
import tensorflow as tf
import sys
import pathlib
import random
sys.path.append(str(pathlib.Path(__file__).parents[2]))

"""
Sequence are a safer way to do multiprocessing. This structure guarantees that the network will only train 
once on each sample per epoch which is not the case with generators.

https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
"""

logger = generate_logger(__name__, "main.log")


class DataGeneratorV2(tf.keras.utils.Sequence):

    """
      Generator: 
        - gets file names depending on train/val/test
        - opens the files and prepares a batch (list of raw data)
        - calls the corresponding preprocessors
        - returns the data to be fed into the model 
          - ([encoder_input, decoder_input], decoder_output)
    """

    def __init__(self, mode, split, batch_size, one_hot=True):
        """
          mode: "train", "validation", "test"
        """

        self.mode = mode
        self.split = split
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.audio_files, self.text_files = self.get_file_names(mode, split)

    def process_audio_files(self, index):
        audio_data = []
        for file_path in self.audio_files[index * self.batch_size:(index + 1) * self.batch_size]:
            audio_data.append(load_audio(file_path))

        encoder_input = PreprocessAudioLayer().call(audio_data)
        return encoder_input

    def process_text_files(self, index):
        text_data = []
        for file_path in self.text_files[index * self.batch_size:(index + 1) * self.batch_size]:
            with open(file_path, "r") as f:
                text_data.append(f.read())

        decoder_input, decoder_output = PreprocessTextLayer().call(text_data)
        return decoder_input, decoder_output

    def on_epoch_end(self):
        """
          shuffle the dataset at the end of each epoch
        """
        random.seed(42)
        random.shuffle(self.audio_files)
        random.seed(42)
        random.shuffle(self.text_files)

    def __len__(self):
        """
          returns the number of batches in the sequence 
            - return value must be of type int
        """
        return len(self.audio_files) // self.batch_size

    def __getitem__(self, index):
        """
          index - passed in by training model

          returns batch, (input, output) value pair, at position index
            - returned values must be numpy arrays
        """
        encoder_input = self.process_audio_files(index)
        decoder_input, decoder_output = self.process_text_files(index)
        logger.info(
            f"Shapes: {encoder_input.shape, decoder_input.shape, decoder_output.shape}")
        return (encoder_input, decoder_input), decoder_output

    @staticmethod
    def get_file_names(mode, split):

        def fetch_names(audio_path, text_path):
            audio_files = [file.path for file in os.scandir(
                os.path.abspath(audio_path))]
            text_files = [file.path for file in os.scandir(
                os.path.abspath(text_path))]

            assert len(audio_files) == len(
                text_files), f"Number of files must be equal: Audio {len(audio_files)}, Text {len(text_files)}"
            logger.info(f"{mode} set length: {len(audio_files)}")

            return audio_files, text_files

        if mode == "train":
            audio_files, text_files = fetch_names(
                config.AUTIO_TRAIN_PATH, config.TEXT_TRAIN_PATH)
            split_index = int(np.floor(len(audio_files) * split))
            return audio_files[:split_index], text_files[:split_index]

        if mode == "validation":
            audio_files, text_files = fetch_names(
                config.AUTIO_TRAIN_PATH, config.TEXT_TRAIN_PATH)
            split_index = int(np.floor(len(audio_files) * split))
            return audio_files[split_index:], text_files[split_index:]

        if mode == "test":
            return fetch_names(config.AUDIO_TEST_PATH, config.TEXT_TEST_PATH)
