import pandas as pd
import numpy as np
from os import path, scandir, makedirs

file_dir = "output/"
output_dir = "output/synthetic_dt/"

def process_dt_data(dt_csv, sentence_csv):
    sentence = pd.read_csv(sentence_csv).columns[5:]
    names = [f"{i + 1}_{word}" for i, word in enumerate(sentence) if word != '""']

    dt_df = pd.read_csv(dt_csv, names=names)
    average = dt_df.apply(np.mean)
    return pd.DataFrame(average)




dirs = [f"{item.name}/" for item in scandir(file_dir) if path.isdir(item)]

for current_dir in dirs[0:1]:
    dir_path = f"{file_dir}/{current_dir}"
    models = [model.name for model in scandir(dir_path) if path.isdir(model)]

    for model in models[0:1]:
        ave_dt_data = []

        model_path = f"{dir_path}/{model}/"
        dt_data = [f"{model_path}{file.name}" for file in scandir(model_path) if "DT" in file.name]
        sentence_data = [f"{model_path}{file.name}" for file in scandir(model_path) if "DT" not in file.name]

        for i, dt_csv in enumerate(dt_data):
            sentence_csv = sentence_data[i]
            assert dt_csv[-14:] == sentence_csv[-14:]

            ave_dt_data.append(process_dt_data(dt_csv, sentence_csv))

        sentence_df = pd.concat(ave_dt_data)
        sentence_df = sentence_df.reset_index()
        sentence_df.columns = ["word", "mean_dt"]
        sentence_df["word"] = sentence_df["word"].apply(lambda s: s.split("_")[1])

        output_dir = f"{output_dir}/{current_dir}/"

        if not path.exists(output_dir):
            makedirs(output_dir)

        sentence_df.to_csv(f"{output_dir}/{model}.csv", index=False)