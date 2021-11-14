import argparse
from models.lstm import VanillaLSTM
from models.lstm_v2 import VanillaLSTMV2
from utils.model_document_object import ModelDocumentObject
import config

parser = argparse.ArgumentParser()
parser.add_argument(dest="mode", choices=[
                    "train", "test", "predict"], help="Select which mode to run in, train, test or predict")
parser.add_argument("--model", type=str, help="Choose a model")
parser.add_argument("--load", help="Choose version number or type latest")
parser.add_argument("--debug", help="Puts the program in debug mode", action="store_true")

models = {
    "vanilla_lstm": VanillaLSTM,
    "vanilla_lstm_v2": VanillaLSTMV2
}

params = {
    "audio_dim": 26,
    "hidden_dim": 32,
    "epochs": 2,
    "batch_size": 64,
    "split": 0.9,
    "one_hot": True,
    "data_source": "generator",
    "metrics": ["accuracy"],
    "optimizer": "rmsprop",
    "loss": "categorical_crossentropy"
}

if __name__ == "__main__":
    args = parser.parse_args()

    with ModelDocumentObject() as document_object:
        config.init_document(document_object)
        config.DEBUG_MODE = args.debug
        
        if args.mode == "train":
            model = models[args.model](mode=args.mode, load_version=args.load, **params)
            model.train()
            pass

        if args.mode == "test":
            model = models[args.model](mode=args.mode, load_version=args.load)
            model.test()
            pass

        if args.mode == "predict":
            model = models[args.model](mode=args.mode, load_version=args.load)
            model.inference()
            pass
