import numpy as np
import pandas as pd

from models.swift_2 import calculate_process_rate, activation_change, check_for_saccade, get_saccade_delta
from models.utils import find_fixated_word
from preprocessing.dataframe import process_df
from preprocessing.swift_2 import process_sentence

max_time = 10000  # ms
time_step = 2  # ms
simulation_steps = int(max_time / time_step)


def metadata_step(trial_id, i, k, word_pos):
    timestamp = i * time_step
    fixated_word = find_fixated_word(k, word_pos)

    return np.array((trial_id, timestamp, k, fixated_word), dtype="int")


def trial_step(current_processed, k, word_complete, difficulty, word_pos, word_prob):
    word_rates = calculate_process_rate(k, word_pos)
    processed, complete = activation_change(current_processed, word_complete, difficulty, word_pos, word_prob,
                                            word_rates, time_step=time_step)

    return complete, processed


def saccade_step():
    pass


def run(trial_id, df):
    num_words = df.shape[0]
    k = 0
    prev_saccade = 0

    saccade_delta = get_saccade_delta()
    word_complete, difficulty, word_pos, word_prob = process_sentence(df)

    trial = np.zeros((simulation_steps, num_words))
    meta_data = np.zeros((simulation_steps, 4), dtype="int")  # trial_id, timestamp, k, fixated_word

    meta_data[0, :] = np.array((trial_id, 0, k, 0), dtype="int")

    for i in range(1, simulation_steps):
        timestamp = i * time_step

        meta_data[i, :] = metadata_step(trial_id, timestamp, k, word_pos)
        word_complete, trial[i, :] = trial_step(trial[i - 1, :], k, word_complete, difficulty, word_pos, word_prob)

        if check_for_saccade(timestamp, prev_saccade, saccade_delta, trial[i, meta_data[i, 3]]):
            saccade_step()
            prev_saccade = timestamp
            saccade_delta = get_saccade_delta()

    return np.concatenate([meta_data, trial], axis=1)


def test():
    data = pd.read_csv("models/data/EZ-corpus.csv", header=None, names=["freq", "len", "prob", "word"])
    test_data = data.loc[:11].copy()

    test_data = process_df(test_data)

    print(run(1, test_data))


if __name__ == "__main__":
    test()
