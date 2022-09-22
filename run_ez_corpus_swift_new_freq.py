import pandas as pd
from os import path, makedirs
from preprocessing.dataframe import partition_sentences, get_freq_per_million
import training_loops

output_dir = "output/ez-corpus_v2/"

def main():
    data = pd.read_csv("data/EZ-corpus.csv", header=None, names=["freq", "len", "prob", "word"])
    data = data.drop("freq", axis=1)
    data["freq"] = data["word"].apply(get_freq_per_million)

    if not path.exists(output_dir):
        makedirs(output_dir)

    sentences = partition_sentences(data)

    training_loops.main(data, sentences, output_dir)


if __name__ == "__main__":
    main()
