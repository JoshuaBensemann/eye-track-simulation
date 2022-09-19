import numpy as np
import pandas as pd

from models.swift_2 import calculate_process_rate, activation_change, check_for_labile_start, get_saccade_delta, \
    calculate_activation_integral, start_labile_stage, start_nonlabile_stage, start_saccade_execution
from models.utils import find_fixated_word, find_word_centre
from preprocessing.dataframe import swift_2_process_df
from preprocessing.swift_2 import process_sentence

max_time = 10000  # ms
time_step = 2  # ms
simulation_steps = int(max_time / time_step)


def metadata_step(trial_id, i, k, word_pos):
    timestamp = i * time_step
    fixated_word = find_fixated_word(k, word_pos)

    return np.array((trial_id, timestamp, k, fixated_word, 0))


def trial_step(current_processed, k, word_complete, difficulty, word_pos, word_prob, suppression):
    word_rates = calculate_process_rate(k, word_pos)
    processed, complete = activation_change(current_processed, k, word_complete, difficulty, word_pos, word_prob,
                                            word_rates, time_step, suppression)

    return complete, processed


def labile_step(time_stamp):
    labile_timer = start_labile_stage() + time_stamp
    prev_saccade_t = time_stamp
    saccade_delta = get_saccade_delta()

    return labile_timer, prev_saccade_t, saccade_delta


def nonlabile_step(time_stamp, current_activations, k, word_pos, current_word):
    target_word, nonlabile_timer, post_saccade_k = start_nonlabile_stage(current_activations, k, word_pos, current_word)
    nonlabile_timer = nonlabile_timer + time_stamp

    return target_word, nonlabile_timer, post_saccade_k


def saccade_step(time_stamp, word_pos, k, current_word, target_word):
    execution_timer = start_saccade_execution()
    execution_timer = execution_timer + time_stamp
    suppression_start = time_stamp
    suppression_stop = execution_timer

    return execution_timer, suppression_start, suppression_stop


def post_saccade_step(time_stamp, post_saccade_k, target_word, word_pos):
    if find_fixated_word(post_saccade_k, word_pos) == target_word:
        saccade_delta = get_saccade_delta()
    else:
        saccade_delta = 0  # missed target

    labile_timer = 0
    nonlabile_timer = 0
    execution_timer = 0
    k = post_saccade_k
    prev_saccade_t = time_stamp

    return saccade_delta, labile_timer, nonlabile_timer, execution_timer, k, prev_saccade_t


def check_for_trial_end(word_complete, word_prob):
    for i in range(word_complete.shape[0]):
        if word_complete[i] == 0 and word_prob[i] < 1:
            return False

    return True


def run(trial_id, df):
    num_words = df.shape[0]
    post_saccade_k = 0
    prev_saccade_t = 0
    labile_timer = 0
    nonlabile_timer = 0
    execution_timer = 0
    suppression_start = 0
    suppression_stop = 0
    target_word = 0

    saccade_delta = get_saccade_delta()
    word_complete, difficulty, word_pos, word_prob = process_sentence(df)
    k = find_word_centre(word_pos[target_word, :])

    trial = np.zeros((simulation_steps, num_words))
    meta_data = np.zeros((simulation_steps, 5))  # trial_id, time_stamp, k, fixated_word, event
    dwell_time = np.zeros(num_words)

    meta_data[0, :] = np.array((trial_id, 0, k, 0, 0))

    for i in range(1, simulation_steps):
        time_stamp = i * time_step

        meta_data[i, :] = metadata_step(trial_id, i, k, word_pos)
        current_word = int(meta_data[i, 3])

        suppression = suppression_start <= time_stamp <= suppression_stop

        if not suppression and current_word < dwell_time.shape[0]:
            dwell_time[current_word] = dwell_time[current_word] + time_step # add dwell time unless eye is moving

        word_complete, trial[i, :] = trial_step(trial[i - 1, :], k, word_complete, difficulty, word_pos, word_prob,
                                                suppression)

        if nonlabile_timer == 0:  # at point of no return if > 0
            if current_word == -1 or current_word == word_pos.shape[0]:
                activation_integral = 0
            else:
                activation_integral = calculate_activation_integral(trial[:i + 1, current_word], time_stamp, time_step)

            if check_for_labile_start(time_stamp, prev_saccade_t, saccade_delta, activation_integral):
                labile_timer, prev_saccade_t, saccade_delta = labile_step(time_stamp)
                meta_data[i, 4] = 1  # started labile phase

            if time_stamp >= labile_timer > 0:
                target_word, nonlabile_timer, post_saccade_k = nonlabile_step(time_stamp, trial[i, :],
                                                                              k, word_pos, current_word)
                meta_data[i, 4] = 2  # started nonlabile phase

        else:
            if time_stamp >= nonlabile_timer and execution_timer == 0:
                execution_timer, suppression_start, suppression_stop = \
                    saccade_step(time_stamp, word_pos, k, current_word, target_word)
                meta_data[i, 4] = 3 # started saccade

            if time_stamp >= execution_timer > 0:
                saccade_delta, labile_timer, nonlabile_timer, execution_timer, k, prev_saccade_t = \
                    post_saccade_step(time_stamp, post_saccade_k, target_word, word_pos)
                meta_data[i, 4] = 4 # end of saccade
                if check_for_trial_end(word_complete, word_prob):
                    if current_word >= word_complete.shape[0] - 2:
                        break

    full_data = np.concatenate([meta_data, trial], axis=1)
    word_series = df["word"].squeeze()
    trial_cols = ["trial_id", "time_stamp", "k", "fixated_word", "event"] + word_series.tolist()
    trial_df = pd.DataFrame(full_data[:i+1, :], columns=trial_cols)

    dwell_time_series = pd.Series(dwell_time)

    return trial_df, dwell_time_series


def test():
    data = pd.read_csv("data/EZ-corpus.csv", header=None, names=["freq", "len", "prob", "word"])
    test_data = data.loc[:11].copy()

    test_data = swift_2_process_df(test_data)

    trial, dwell_time = run(1, test_data)
    print("Done")


if __name__ == "__main__":
    test()
