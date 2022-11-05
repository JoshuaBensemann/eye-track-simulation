import pandas as pd
from os import path, makedirs
from preprocessing.dataframe import partition_sentences, get_freq_per_million
import training_loops

output_dir = "output/provo/real/"


def main():
    data = pd.read_csv("data/csv/provo_data.csv")
    data["freq"] = data["word"].apply(get_freq_per_million)

    if not path.exists(output_dir):
        makedirs(output_dir)

    sentences = partition_sentences(data)

    training_loops.main(data, sentences, output_dir)


if __name__ == "__main__":
    main()
