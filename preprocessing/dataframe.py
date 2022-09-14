import pandas as pd
from preprocessing.swift import max_activation


def process_df(df):
    df["word"] = df["word"].apply(lambda s: s.split(".")[0])

    start = 0
    stop = 0
    word = 1

    data_dict = {}

    for i, row in df.iterrows():
        stop = start + row["len"] - 1
        data_dict[i] = {"word": word, "start": start, "stop": stop}

        word = word + 1
        start = stop + 2

    word_info = pd.DataFrame.from_dict(data_dict, orient="index")

    df = pd.concat([df, word_info], axis=1)
    df["ln"] = df["freq"].apply(max_activation)
    return df
