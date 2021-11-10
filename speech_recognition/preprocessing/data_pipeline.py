import tensorflow as tf
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config

"""
  1. does dataset.map() return a generator? If not how does it perform operations on predefined batches only?
      - might do it on the first batch and wait for fit() method to call on next batch?
"""
def process_audio_file_path(file_path):
  audio = tf.io.read_file(file_path)
  return audio

def preprocess_audio_text_path(file_path):
  text = tf.io.read_file(file_path)
  return input_text, output_text

num_of_samples = 10
audio_files = tf.data.Dataset.list_files(f"{config.AUTIO_TRAIN_PATH}/*")
text_files = tf.data.Dataset.list_files(f"{config.TEXT_TRAIN_PATH}/*")

dataset = tf.data.Dataset.zip((audio_files, text_files))
dataset.shuffle(buffer_size=len(audio_files), reshuffle_each_iteration=True)

split = 0.8
train_ds = dataset.take(int(len(dataset) * split))
val_ds = dataset.skip(split)

train_ds = train_ds.batch(batch_size=128)
val_ds = val_ds.batch(batch_size=128)
# do preprocessing