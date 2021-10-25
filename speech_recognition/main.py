import argparse
from models.lstm import VanillaLSTM
from models.model_document_object import ModelDocumentObject
import config

parser = argparse.ArgumentParser()
parser.add_argument(dest="mode", choices=["train", "test", "predict"], help="Select which mode to run in, train, test or predict")
parser.add_argument("--model", type=str, help="Choose a model")
parser.add_argument("--load", help="Choose version number or type latest")

models = {
  "vanilla_lstm": VanillaLSTM
}

if __name__ == "__main__":
  args = parser.parse_args()

  with ModelDocumentObject() as document_object:
    config.init_document(document_object)
    
    if args.mode == "train":
      model = models[args.model](load_version=args.load)
      model.fit()
      pass

    if args.mode == "test":
      model = models[args.model](load_version=args.load)
      model.test()
      pass

    if args.mode == "predict":
      model = models[args.model](load_version=args.load)
      model.predict()
      pass