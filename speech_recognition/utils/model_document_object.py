import sys
import pathlib
import os
import json
sys.path.append(str(pathlib.Path(__file__).parents[1]))
import config

"""
  Document = {
    model_name: {
      1: {
        train: {},
        test: {}
      }
      2: {},
      latest_version: 2,
      best_score: 0.8,
      best_version: 2,
      ...
    }
  }
"""

class ModelDocumentObject():

  # CHANGE PATH
  # str(pathlib.Path(__file__).parents[2]) + "/models/" + model_name
  FILE_PATH = str(pathlib.Path(__file__).parent) + "/" + "model_document_object.json"

  def __init__(self):
    self.document = None

  def __enter__(self):
    if not os.path.exists(self.FILE_PATH):
      self.document = {}

    else:
      with open(self.FILE_PATH) as file:
        self.document = json.load(file)
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    """
      If anything other than True is returned the with statement will throw an exception
    """

    with open(self.FILE_PATH, "w") as file:
      json.dump(self.document, file)

  def add_model(self, model_name):
    if not model_name in self.document:
      self.document[model_name] = {"latest_version": 1}
      self.document[model_name]["1"] = {
        "train": {},
        "test": {}
      }

  def check_version_exists(self, model_name, version):
    try:
      self.document[model_name][version]
      return True
    except:
      return False

  def get_latest_version(self, model_name):
    try:
      return self.document[model_name]["latest_version"]
    except:
      return 0

  def update_version(self, model_name):
    try:
      version = self.document[model_name]["latest_version"] + 1
      self.document[model_name]["latest_version"] = version
      self.document[model_name][version] = {
        "train": {},
        "test": {}
      }
      return self.document[model_name]["latest_version"]
    except:
      return 0

  def append_metric(self, model_name, version, stage, metric, value):
    version = str(version)
    try:
      self.document[model_name][version][stage][metric].append(value)
    except:
      self.document[model_name][version][stage][metric] = [value]

  def add_field(self, **kwargs):
    for kwarg in kwargs:
      self.document[kwarg] = kwargs[kwarg]
    