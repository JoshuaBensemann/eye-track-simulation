import pandas as pd
from os import path, makedirs

import swift_2
from preprocessing.dataframe import partition_sentences, swift_2_process_df
from multiprocessing.pool import Pool

trials = 200
output_dir = "output/ez-corpus/"


def run_trial(sentence_id, trial_id, df):
    trial, dwell_time = swift_2.run(trial_id, df)
    print(f"Sentence {sentence_id} Trial {trial_id} complete")

    return trial, dwell_time


def main():
    data = pd.read_csv("data/EZ-corpus.csv", header=None, names=["freq", "len", "prob", "word"])
    if not path.exists(output_dir):
        makedirs(output_dir)

    sentences = partition_sentences(data)

    for sentence_id, sentence in enumerate(sentences):
        with Pool() as pool:
            trial_results = []
            dwell_time_results = []

            start = sentence[0]
            stop = sentence[1]
            current_data = data.iloc[start:stop].copy()
            current_data = swift_2_process_df(current_data)

            items = [(sentence_id, trial_id, current_data) for trial_id in range(trials)]
            results = pool.starmap(run_trial, items)

            for trial, dwell_time in results:
                trial_results.append(trial)
                dwell_time_results.append(dwell_time)

            full_trial_results = pd.concat(trial_results)
            full_dwell_time_results = pd.concat(dwell_time_results, axis=1)

            full_trial_results.to_csv(f"{output_dir}Sentence_{sentence_id}.csv", index=False)
            full_dwell_time_results.transpose().to_csv(f"{output_dir}DT_Sentence_{sentence_id}.csv", index=False)

            del full_dwell_time_results, full_trial_results, trial_results, dwell_time_results


if __name__ == "__main__":
    main()