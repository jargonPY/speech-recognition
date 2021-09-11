import os
import pathlib

dir = pathlib.Path(__file__).parents[2]
audio_path = str(dir) + "/data/audio_files"
text_path = str(dir) + "/data/text_files"
dataset_path = str(dir) + "/data/datasets/archive/data"

def move_timit():
    
    if not os.path.isdir(audio_path):
      print("HERERERERE")
      os.mkdir(audio_path)
    if not os.path.isdir(text_path):
      os.mkdir(text_path)

    for folder in ["TRAIN", "TEST"]:
      path = dataset_path + "/" + folder
      num_files = 0
      for root, subdir, files in os.walk(path):
        text_file_name = ""
        for filename in files:
          if "WAV.wav" in filename:
            #if " " in filename.split(".")[0]: continue
            text_file_name = filename.split(".")[0] + ".TXT"
            os.rename(root + "/" + filename, audio_path + f"/{folder.lower()}/audio_{num_files}.wav")
            os.rename(root + "/" + text_file_name, text_path + f"/{folder.lower()}/text_{num_files}.txt")
            num_files += 1

    print("Audio train length: ", len(os.listdir(audio_path + "/train")))
    print("Text train length: ", len(os.listdir(text_path + "/train")))

    print("Audio test length: ", len(os.listdir(audio_path + "/test")))
    print("Text test length: ", len(os.listdir(text_path + "/test")))

if __name__ == "__main__":
    move_timit()