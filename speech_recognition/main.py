import argparse
from models.lstm import VanillaLSTM

parser = argparse.ArgumentParser()
parser.add_argument(dest="mode", choices=["train", "test", "predict"], help="This is the first argument")
parser.add_argument("--model", type=str, help="This is the first argument")
parser.add_argument("--version", type=int, help="This is the first argument")
parser.add_argument("--load_model", default=False)

models = {
  "vanilla_lstm": VanillaLSTM
}

if __name__ == "__main__":
  args = parser.parse_args()
  print("ARGS: ", args)
  
  if args.mode == "train":
    model = models[args.model](load_model=args.load_model, version=args.version)
    pass

  if args.mode == "test":
    model = models[args.model](load_model=True, version=args.version)
    pass

  if args.mode == "predict":
    model = models[args.model](load_model=True, version=args.version)
    pass

# if mode_run == "train":
#   train_set, val_set = get_file_names(split=0.9)
#   train_generator = DataGenerator(train_set, 128, one_hot=True)
#   val_generator = DataGenerator(val_set, 128, one_hot=True)
#   history = model.fit(train_generator,
#                       validation_data=val_generator,
#                       epochs=1,
#                       #max_queue_size=1,
#                       workers=6,
#                       use_multiprocessing=True)
# tf.keras.utils.plot_model(model, show_shapes=True)

# if mode_run == "test":
#   pass
