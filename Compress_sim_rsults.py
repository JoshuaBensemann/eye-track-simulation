import pandas as pd
import numpy as np
import re

sentence_data = {}
trace_data = []
num_sentence = 0

with open(sample, "r") as f:
    for line in f:
        if "dur:" not in line:
            is_sentence = re.search("Sentence +(\d+)", line)
            if is_sentence:
                num_sentence = int(is_sentence.group(1))
                sentence_data[num_sentence] = []

            else:
                line = line.lstrip()
                line = line.rstrip("\n* ")
                line = line.split()
                sentence_data[num_sentence].append(line[-1])

        else:
            trace_data.append(line.split())
            line = f.readline()

    trace_df = pd.DataFrame(trace_data).drop([0, 2, 3, 4], axis=1)
    trace_df.columns = ["duration", "word_num", "word"]
    trace_df['duration'] = trace_df['duration'].astype(int)
    trace_df['word_num'] = trace_df['word_num'].astype(int)

current_sentence = 0
current_word = 0
current_participant = 0
trace_df["sentence"] = 0
trace_df["participant"] = 0

for i, row in trace_df.iterrows():
    word_num = int(row["word_num"])
    word = row["word"]

    if sentence_data[current_sentence][word_num] != word:
        current_sentence = current_sentence + 1
        if current_sentence == len(sentence_data):
            current_sentence = 0
            current_participant = current_participant + 1

    assert sentence_data[current_sentence][word_num] == word
    trace_df.loc[i, 'participant'] = current_participant
    trace_df.loc[i, 'sentence'] = current_sentence

trace_df.to_csv("frank_bert-base-cased_raw.csv", index=False)

