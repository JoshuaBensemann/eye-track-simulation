import pandas as pd
import re
from os import path, makedirs, scandir

output_dir = "output/ez_sims/"

if not path.exists(output_dir):
    makedirs(output_dir)


def compress_data(input_file):
    filename = input_file.split("/")[-1][:-4]
    print(filename)
    sentence_data = {}
    trace_data = []
    num_sentence = 0

    print("Reading", end="...")
    with open(input_file, "r") as f:
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

    trace_df = pd.DataFrame(trace_data).drop([0, 2, 3, 4], axis=1)
    trace_df.columns = ["duration", "word_num", "word"]
    trace_df['duration'] = trace_df['duration'].astype(int)
    trace_df['word_num'] = trace_df['word_num'].astype(int)

    current_sentence = 0
    current_participant = 0
    trace_df["sentence"] = 0
    trace_df["participant"] = 0
    df_len = trace_df.shape[0]

    print("Done!")

    print(f"Converting {df_len} rows")

    for i, row in trace_df.iterrows():
        if i%10000 == 0:
            print(f"Completed {i} rows of {df_len} - {(i/df_len)*100:.2f}%")
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

    trace_df.to_csv(f"{output_dir}{filename}.csv", index=False)

    print("Done!")


def main():
    print("Enter data directory:", end=" ")
    input_dir = input()

    sim_files = [f"{input_dir}{file.name}" for file in scandir(input_dir) if "SimulationResults" in file.name]

    for sim_file in sim_files:
        compress_data(sim_file)

    print("All files have been processed")


if __name__ == "__main__":
    main()
