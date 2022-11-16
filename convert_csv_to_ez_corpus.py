from os import makedirs, path, scandir

import pandas as pd

from preprocessing.dataframe import get_freq_per_million

input_dir = "data/csv/"
output_dir = "data/ez_format/"

if not path.exists(output_dir):
    makedirs(output_dir)


def df_to_line(row):
    line = f'{int(row["freq"])}\t{row["len"]}\t{row["prob"]:.2f}\t{row["word"]}\n'
    return line


def find_end_sentence(word):
    try:
        if word[-1] in [".", "?", "!"]:
            word = f"{word}@"
    except Exception:
        word = "null"

    return word


def convert_to_ez_format(csv_file):
    df = pd.read_csv(f"{input_dir}{csv_file}")
    if "freq" not in df.columns:
        df["freq"] = df["word"].apply(get_freq_per_million)

    df["word"] = df["word"].apply(find_end_sentence)
    lines = df.apply(df_to_line, axis=1)

    return lines.tolist()


def create_corpus_and_targets(csv_file, converted):
    sentences = 0

    output_file = csv_file.replace(".csv", ".txt")

    with open(f"{output_dir}{output_file}", "w", encoding="utf-8") as f:
        for line in converted:
            if "@" in line:
                sentences = sentences + 1
            f.write(line)

    with open(f"{output_dir}targets_{output_file}", "w", encoding="utf-8") as f:
        for i in range(sentences):
            f.write("0\n")


files = [file.name for file in scandir(input_dir)]

for csv_file in files:
    print(csv_file)
    converted = convert_to_ez_format(csv_file)
    create_corpus_and_targets(csv_file, converted)
