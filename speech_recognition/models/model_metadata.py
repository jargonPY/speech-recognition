import sys
import pathlib
import os
sys.path.append(str(pathlib.Path(__file__).parents[2]))
import speech_recognition.config as config
import json

"""
  MODELS = {
    model_name: {
      version: 4,
      best_score: 0.8,
      best_version: 2
      ...
    }
  }
"""
class ModelsMetadata():

  FILE_PATH = str(pathlib.Path(__file__).parent) + "/" + "models_metadata.json"

  def __init__(self):
    if not os.path.exists(ModelsMetadata.FILE_PATH):      
      self.models_metadata = {}
    
    else:
      with open(ModelsMetadata.FILE_PATH) as f:
        self.models_metadata = json.load(f)

  def check_version_exists(self, model_name, version):
    return self.models_metadata[model_name]["latest_version"] >= version

  def get_latest_model_version(self, model_name):
    if model_name in self.models_metadata: # should all functions have checks?
      return self.models_metadata[model_name]["latest_version"]

  def update_model_version(self, model_name):
    self.models_metadata[model_name]["latest_version"] += 1
    with open(ModelsMetadata.FILE_PATH, "w") as f:
      json.dump(self.models_metadata, f)
    return self.models_metadata[model_name]["latest_version"]

  def add_new_model(self, model_name):
    self.models_metadata[model_name] = {"latest_version": 0}
    with open(ModelsMetadata.FILE_PATH, "w") as f:
      json.dump(self.models_metadata, f)
    return 0

  def append_training_metric(self, model_name, model_version, metric, value):
    pass
    #self.models_metadata[model_name][f"version_{model_version}"]["train"][metric].append(value)
    # Need to check if field exists (use on_train_begin in callback?)
    # Use context manager to avoid loading/saving json with every call
    # Find a way for model_name and model_version to be included