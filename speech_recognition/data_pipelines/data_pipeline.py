import speech_recognition.config as config
import tensorflow as tf
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parents[2]))

"""
  Pipeline:
   1. get audio and text files
   2. combine into one dataset where each element is: (audio_file, text_file)
   3. split into train and valdiation sets
   4. shuffle (before loading data)
   5. preprocess individual elements
   6. batch
   7. preprocess batch (specifically padding)
"""


def process_audio_file(file_path):
    audio = tf.io.read_file(file_path)
    return audio


def preprocess_text_file(file_path):
    text = tf.io.read_file(file_path)
    return text, text


def preprocess_files(audio_file, text_file):
    audio = process_audio_file(audio_file)
    input_text, output_text = preprocess_text_file(text_file)
    return audio, input_text, output_text


def preprocess_audio_batch(batch):
    pass


def preprocess_text_batch(batch):
    pass


def data_pipeline(split, batch_size, num_of_samples=4):
    """
      Consider processing audio and text as seperate datasets and at the end combining them.
      This might lead to better parallel processing.
    """

    # get file names - returns a Dataset of strings corresponding to file names
    audio_files = tf.data.Dataset.list_files(
        f"{config.AUTIO_TRAIN_PATH}/*", shuffle=False).take(num_of_samples)
    text_files = tf.data.Dataset.list_files(
        f"{config.TEXT_TRAIN_PATH}/*", shuffle=False).take(num_of_samples)

    dataset = tf.data.Dataset.zip((audio_files, text_files))

    # split into train and validations sets
    train_ds = dataset.take(int(len(audio_files) * split))
    val_ds = dataset.skip(int(len(audio_files) * split))

    # shuffle sets
    train_ds = train_ds.shuffle(buffer_size=len(
        audio_files), reshuffle_each_iteration=True)
    val_ds = val_ds.shuffle(buffer_size=len(
        audio_files), reshuffle_each_iteration=True)

    # preprocess single files --> each file is lazly generated and represents a single
    # tensor, before batching the data needs to be padded
    train_ds = train_ds.map(
        lambda audio_file, text_file: preprocess_files(audio_file, text_file),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    val_ds = val_ds.map(
        lambda audio_file, text_file: preprocess_files(audio_file, text_file),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    for train_sample in train_ds:
        print("Train sample: \n", train_sample)

    train_ds = train_ds.padded_batch(
        batch_size=batch_size,
        padding_values="<pad>"
    )

    val_ds = val_ds.padded_batch(
        batch_size=batch_size,
        padding_values="<pad>"
    )
    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = data_pipeline()
    for train_batch in train_ds:
        print("Batch: ", train_batch)
