
#----------------------------- Config --------------------------------------#
AUTIO_TRAIN_PATH = "data/audio_files/train"
TEXT_TRAIN_PATH = "data/text_files/train"

AUDIO_TEST_PATH = "data/audio_files/test"
TEXT_TEST_PATH = "data/text_files/test"


NUM_CLASSES = 30
MAX_DECODER_SEQ_LENGTH = 100

# Cant start at zero, since zero is used for zero padding
TOKEN_TO_INDEX = {'<pad>': 0, '<sos>': 1, '<eos>': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7,
            'f': 8, 'g': 9, 'h': 10, 'i': 11, 'j': 12, 'k': 13, 'l': 14, 
            'm': 15, 'n': 16, 'o': 17, 'p': 18, 'q': 19, 'r': 20, 's': 21, 
            't': 22, 'u': 23, 'v': 24, 'w': 25, 'x': 26, 'y': 27, 'z': 28, '<space>': 29}

INDEX_TO_TOKEN = {
  1: '<sos>', 2: '<eos>', 3: 'a', 4: 'b', 5: 'c', 6: 'd', 7: 'e', 8: 'f',
  9: 'g', 10: 'h', 11: 'i', 12: 'j', 13: 'k', 14: 'l', 15: 'm', 16: 'n',
  17: 'o', 18: 'p', 19: 'q', 20: 'r', 21: 's', 22: 't', 23: 'u', 24: 'v',
  25: 'w', 26: 'x', 27: 'y', 28: 'z', 29: '<space>'
}

#-------------------------- Model Document Object --------------------------------#
document = None

def init_document(document_object):
  global document
  document = document_object