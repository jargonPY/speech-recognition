import sys
import pathlib
import os
import numpy as np
import tensorflow as tf
sys.path.append(str(pathlib.Path(__file__).parents[1]))
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import config
import utils
import data_pipelines
from speech_recognition.custom_callbacks import LogsCallback
from speech_recognition.preprocessing import PreprocessAudio


logger = utils.generate_logger(__name__, "main.log")


class BaseModel():

    def __init__(self, model_name, load_version):
        """
          - store version path
          If dir does not exist:
            - create model dir
            - instantiate version (add to config JSON)
            - create version dir
          If dir does exist:
            - if version number specified in arg load version else load latest
        """

        self.model_name = model_name
        self.model_path = str(pathlib.Path(
            __file__).parents[2]) + "/models/" + model_name
        self.version = None
        self.version_path = None

        if load_version:
            if load_version == "latest" and config.document.check_model_exists(model_name):
                self.version = config.document.get_latest_version(model_name)

            elif config.document.check_version_exists(model_name, load_version):
                self.version = load_version

            else:
                logger.error("Version specified does not exist")
                raise ValueError("The version does not exist")
            self.version_path = self.model_path + f"/version_{self.version}"

        else:
            if not os.path.exists(self.model_path):
                # what if the model folder gets deleted but the json
                # file still has records of the model?
                os.mkdir(self.model_path)
                config.document.add_model(model_name)
                self.version = 0

            else:
                self.version = config.document.update_version(model_name)

            self.version_path = self.model_path + f"/version_{self.version}"
            try:
                os.mkdir(self.version_path)
            except:
                if not config.DEBUG_MODE:
                    raise ValueError("Version path already exists")

            logger.info(
                f"model: {self.model_name}-{self.model_path}, version: {self.version}-{self.version_path}")

    def load_model(self):
        
        if os.path.exists(self.version_path):
            return tf.keras.models.load_model(self.version_path)

    def train(self):

        if self.data_source == "generator":
            train = data_pipelines.DataGeneratorV2("train", self.split, self.batch_size)
            val = data_pipelines.DataGeneratorV2(
                "validation", self.split, self.batch_size)
        if self.data_source == "pipeline":
            train, val = data_pipelines.data_pipeline(self.split, self.batch_size)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.version_path,
            save_freq="epoch"
        )

        logsCallback = LogsCallback(self.model_name, self.version)

        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        history = self.model.fit(train,
                                 validation_data=val,
                                 epochs=self.epochs,
                                 max_queue_size=1,
                                 workers=1,
                                 use_multiprocessing=True,
                                 callbacks=[checkpoint, logsCallback])
        return history

    def test(self):
        # combine inference and evaluation
        test_generator = data_pipelines.DataGeneratorV2("test", split=0.9, batch_size=1)
        for index in range(test_generator.__len__()):
            input_data, decoder_output = test_generator.__getitem__(index)
            prediction = self.inference(input_data[0])

    def evaluate(self):
        pass

    def inference(self, audio_input):
        """
          For inference the encoder model and the decoder model could make
          the preprocessing layers part of the model
        """

        # audio_input = PreprocessAudio().preprocess_file(audio_input)
        # audio_input = np.reshape(audio_input, (1, -1, audio_input.shape[1]))
        encoder_state = self.encoder_model.predict(audio_input)

        # (1 sample, 1 timestep, NUM_CLASSES)
        target_seq = np.zeros((1, 1, config.NUM_CLASSES))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, config.TOKEN_TO_INDEX["<sos>"]] = 1.0

        # Sampling loop for a batch of sequences (here we assume a batch size of 1)
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            # internal_state = [hidden_state, cell_state]
            output_tokens, internal_state = self.decoder_model.predict(
                [target_seq] + encoder_state)

            # Sample a token (output_tokens.shape = (1,1,30))
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = config.INDEX_TO_TOKEN[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length or find stop character
            if sampled_char == "<eos>" or len(decoded_sentence) > config.MAX_DECODER_SEQ_LENGTH:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, config.NUM_CLASSES))
            # sampled_token_index does not start at zero does that matter?
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            encoder_state = internal_state
        print("Decoded sentence: ", decoded_sentence)
