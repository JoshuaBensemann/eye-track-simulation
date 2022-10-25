import pandas as pd
from os import path, makedirs
from preprocessing.dataframe import partition_texts, get_freq_per_million
import training_loops

output_dir = "output/provo_v2/"


def main():
    data = pd.read_csv("data/csv/provo_data.csv")
    data["freq"] = data["word"].apply(get_freq_per_million)

    if not path.exists(output_dir):
        makedirs(output_dir)

    texts = partition_texts(data)

    training_loops.main(data, texts, output_dir)


if __name__ == "__main__":
    main()
