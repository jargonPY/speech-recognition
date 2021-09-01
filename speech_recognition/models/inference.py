import numpy as np
import os
import sys
import pathlib
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[1]))
import config

"""
  1. load model (provide path to model)
  2. provide path to audio to infer from
  3. function to preform inference
    - preprocess audio
    - convert to index and then to letter
  4. allow for real time audio recording and inference
"""

class Inference():

  def __init__():
    pass
  
  def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
      # predict next char
      yhat, h, c = infdec.predict([target_seq] + state)
      # store prediction
      output.append(yhat[0,0,:])
      # update state
      state = [h, c]
      # update target sequence
      target_seq = yhat
    return array(output)
    
  def predict(self, encoder_input):

    # Encode the input as state vectors.
    encoder_state = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, config.NUM_CLASSES))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, config.TOKEN_TO_INDEX] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + encoder_state)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = config.INDEX_TO_TOKEN[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "<eos>" or len(decoded_sentence) > config.MAX_DECODER_SEQ_LENGTH:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, config.NUM_CLASSES))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        encoder_state = [hidden_state, cell_state]