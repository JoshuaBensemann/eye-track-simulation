import pandas as pd
from os import path, makedirs
from preprocessing.dataframe import partition_sentences, get_freq_per_million
import training_loops

input_dir = "data/csv/"


def main():
    with open("texts.txt", "r") as f:
        text_names = [text.replace("\n", "") for text in f.readlines()]

    with open("models.txt", "r") as f:
        models = [model.replace("\n", "") for model in f.readlines()]
        print(models)

    for text in text_names:

        for model in models:
            filename = f"{input_dir}{text}_{model}.csv"
            data = pd.read_csv(filename)

            output_dir = f"output/{text}/{model}/"

            if not path.exists(output_dir):
                makedirs(output_dir)

            sentences = partition_sentences(data)

            training_loops.main(data, sentences, output_dir)


if __name__ == "__main__":
    main()
