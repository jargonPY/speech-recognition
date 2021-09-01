import os
import pandas as pd
import pathlib

dir = pathlib.Path(__file__).parent.parent
audio_path = os.path.join(dir, "/data/audio_files")
text_path = os.path.join(dir, "/data/text_files")

def move_timit():
    
    dataset_path = "../data/datasets/TIMIT/data"
    if not os.path.isdir(audio_path):
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
    print("Audio test length: ", len(os.listdir(audio_path + "/test")))
    print("Text train length: ", len(os.listdir(text_path + "/train")))
    print("Text test length: ", len(os.listdir(text_path + "/train")))

if __name__ == "__main__":
    #move_timit()
    csv = os.path.join(dir, "/data/datasets/TIMIT/train_data.csv")
    csv = dir / "/data/datasets/TIMIT/train_data.csv"
    print("DIR: ", dir)
    print("CSV: ", csv)
    df = pd.read_csv(csv)
    df.head()