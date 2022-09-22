import pandas as pd
from wordfreq import word_frequency

from preprocessing.swift_2 import max_activation


def partition_sentences(df):
    sentence_indices = []
    sentence_start = 0

    for i, row in df.iterrows():
        if row["word"][-1] in [".", "!", "?"]:
            sentence_end = i + 1
            sentence_indices.append([sentence_start, sentence_end])
            sentence_start = sentence_end

    return sentence_indices


def partition_texts(df, id_col="Word_ID"):
    text_indices = []
    text_id = None
    text_start = 0

    for i, row in df.iterrows():
        current_id = row[id_col].split("_")[0]
        if text_id is None:
            text_id = current_id
        else:
            if text_id != current_id:
                text_end = i
                text_indices.append([text_start, text_end])
                text_start = text_end
                text_id = current_id

    return text_indices


def swift_2_process_df(df):
    start = 0
    word = 1

    data_dict = {}

    for i, row in df.iterrows():
        stop = start + row["len"] - 1
        data_dict[i] = {"word_num": word, "start": start, "stop": stop}

        word = word + 1
        start = stop + 2

    word_info = pd.DataFrame.from_dict(data_dict, orient="index")

    df = pd.concat([df, word_info], axis=1)
    df["ln"] = df["freq"].apply(max_activation)
    return df


def get_freq_per_million(word, lang='en', min_value=1):
    min_value = min_value * 1e-6

    return word_frequency(word, lang, minimum=min_value) * 1e6
