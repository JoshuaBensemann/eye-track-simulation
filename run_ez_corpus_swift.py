import pandas as pd
from os import path, makedirs
import training_loops
from preprocessing.dataframe import partition_sentences

output_dir = "output/ez-corpus/"


def main():
    data = pd.read_csv("data/csv/schilling_data.csv", header=None, names=["freq", "len", "prob", "word"])
    if not path.exists(output_dir):
        makedirs(output_dir)

    sentences = partition_sentences(data)

    training_loops.main(data, sentences, output_dir)


if __name__ == "__main__":
    main()
