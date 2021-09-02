import argparse
from models.lstm import VanillaLSTM

parser = argparse.ArgumentParser()
parser.add_argument(dest="mode", choices=["train", "test", "predict"], help="Select which mode to run in, train, test or predict")
parser.add_argument("--model", type=str, help="Choose a model")
parser.add_argument("--version", type=int, help="Choose which version of the model to load")
parser.add_argument("--load_model", default=False)

models = {
  "vanilla_lstm": VanillaLSTM
}

if __name__ == "__main__":
  args = parser.parse_args()
    
  if args.mode == "train":
    model = models[args.model](load_model=args.load_model, version=args.version)
    model.fit()
    pass

  if args.mode == "test":
    model = models[args.model](load_model=True, version=args.version)
    model.test()
    pass

  if args.mode == "predict":
    model = models[args.model](load_model=True, version=args.version)
    model.predict()
    pass