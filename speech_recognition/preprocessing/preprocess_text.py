import string
import sys
import pathlib
import os
import numpy as np
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config

"""
1. Should spaces between words be left as emptry strings?
2. Need to go over all files to add all unique characters or just hardcode
   unique characters to the alphabet and specials ex. <eos>
3. But need to zero pad to the longest token length
4. Also we can either use indices or one hot encoding 
    --> using indices we can specify the number of classes in the embedding layer
"""

"""
How to get max length if we load at runtime?
  - the length need only be the max length in the batch
Where to store max length if we do batch preprocessing?

--> all zero padding should be taken care of generator
"""

class PreprocessText:

  def preprocess_text(self, text_file):

    with open(text_file) as f:
      text = f.read()

    text = self.clean_text(text)
    text = [config.TOKEN_TO_INDEX[char] for char in text]

    return text[:-1], text[1:]

  def batch_preprocess_text(self):
    """
        - open all text files
        - call parse file for each file
        - count the number of tokens to get max length
        - save results in respective folders (decoder input and decoder output)
    """

    file_names = os.listdir(path_to_text_files)
    cleaned_text = []
    
    for index, file in enumerate(file_names):
        with open(path_to_text_files + "/" + file) as f:
            for line in f:
                line = self.parse_file(line)
                cleaned_text.append(line)
        print("Files Cleaned: ", index + 1, "Files Remaining: ", len(file_names) - (index + 1))

    max_length = max([len(txt) for txt in cleaned_text])
    
    for index, text in enumerate(cleaned_text):
        decoder_input, decoder_output = self.pad_file(text, max_length)
        file = file_names[index].split(".")[0]
        np.save(path_to_decoder_input + "/" + "input_" +  file, decoder_input)
        np.save(path_to_decoder_ouput + "/" + "output_" + file, decoder_output)
        print("Files Done: ", index + 1, "Files Remaining: ", len(cleaned_text) - (index + 1))


  def clean_text(self, text):
    """
        - clean text file
        - split into list of chars
        - append <sos> and <eos>
      
    """

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.lower().strip()

    text = ['<space>' if char == ' ' else char for char in text]
    text.insert(0, '<sos>')
    text.append('<eos>')

    return text

  def pad_and_index(self, text, max_length):
    """
        - zero pad to match max length
        - convert to index
        - return padded[:-1] --> decoder input and padded[1:] --> decoder output
    """

    padded = np.zeros(max_length)
    for i, char in enumerate(text):
        padded[i] = config.TOKEN_TO_INDEX[char]

    return padded[:-1], padded[1:]