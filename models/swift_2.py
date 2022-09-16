import numpy as np

from models.utils import get_random_time, calculate_integral, find_word_centre

"""
Swift-2 Constants
"""
vis_span_l = 2.41
vis_span_r = 3.74
word_len_exp = 0.448
global_decay_w = 0.01
preprocess_factor_f = 70.2
processing_noise = 2
inhibition_factor_h = 2.62
lambda_0 = np.sqrt(2 / np.pi) * 1 / (vis_span_l + vis_span_r)  # Equation 5

random_timing = 179.0
labile_stage = 108.0
nonlabile_stage = 6.1
time_delay = 375.7
execution_time = 25.0
suppression_delay = 50
latency_mod_k0 = 105.2
latency_mod_k1 = 0.1

f_saccade_err = 0.41
f_refix_err = 0.49
r_refix_err = -0.5
regression_err = -0.15
f_saccade_range_err = 5.4
f_refix_range_err = 5.7
r_refix_range_err = 4.3
regression_range_err = 10.0

saccade_gaussian_err = 0.87
saccade_random_err = 0.084
"""
Determining Processing Rate
"""


def calculate_letter_lambdas(letter_pos):
    letter_lambdas = np.arange(letter_pos[0], letter_pos[1] + 1, dtype="float")
    num_letters = letter_lambdas.shape[0]

    for i in range(num_letters):  # Equation 3
        vis_span = vis_span_r

        if letter_lambdas[i] < 0:
            vis_span = vis_span_l

        letter_lambdas[i] = lambda_0 * np.exp(-((letter_lambdas[i] ** 2) / 2 * vis_span ** -2))

    return letter_lambdas


def calculate_word_lambda(letter_lambdas):
    word_rate = letter_lambdas.shape[0] ** (-word_len_exp) * np.sum(letter_lambdas)  # Equation 6

    return word_rate


def calculate_process_rate(k, word_pos):
    num_words = word_pos.shape[0]

    word_rates = np.zeros(num_words)

    for i in range(num_words):
        letter_lambdas = calculate_letter_lambdas(word_pos[i, :] - k)  # Equation 2 :word_pos[i,:]-k
        word_rates[i] = calculate_word_lambda(letter_lambdas)

    return word_rates


"""
Calculating Activation
"""


def calculate_preprocessing_factor(word_complete, word_pos, word_prob, suppression):  # Equation 8
    num_words = word_complete.shape[0]
    preprocess_factors = np.zeros(num_words)

    for i in range(num_words):
        prob = word_prob[i]

        if word_complete[i] == 1:  # Lexical completion
            preprocess_factors[i] = -(1 + prob)

        elif suppression:
            preprocess_factors[i] = 0

        elif word_pos[i, 0] <= 1:  # Current fixation is on this word, the space before it, or past this word.
            preprocess_factors[i] = preprocess_factor_f

        else:
            preprocess_factors[i] = preprocess_factor_f * (1 - prob)

    return preprocess_factors


def calculate_stochastic_processing(word_rates):
    num_words = word_rates.shape[0]
    noise = np.random.normal(0, 1, num_words)
    stochastic_rates = word_rates * (1 + processing_noise * noise)  # Equation 9

    return stochastic_rates


def activation_change(current_processed, word_complete, difficulty, word_pos, word_prob, word_rates,
                      time_step, suppression):  # Equation 7
    preprocess_factors = calculate_preprocessing_factor(word_complete, word_pos, word_prob, suppression)
    stochastic_rates = calculate_stochastic_processing(word_rates)
    activations_changes = preprocess_factors * stochastic_rates - global_decay_w

    processed = current_processed.copy()
    complete = word_complete.copy()

    for i in range(current_processed.shape[0]):
        processed[i] = max(processed[i] + time_step * activations_changes[i], 0)
        if processed[i] >= difficulty[i]:
            complete[i] = 1
            processed[i] = difficulty[i]

    return processed, complete


"""
Saccade Programming
"""


def saccade_target_selection(current_activations):
    probs = current_activations / np.sum(current_activations)
    target = np.random.choice(current_activations.shape[0], p=probs)
    return target


def calculate_saccade_amplitude(k, target_pos, refix):
    intended_amplitude = target_pos - k

    if k < target_pos:  # forward saccade/refix
        if refix:
            saccade_err = f_refix_err
            saccade_range_err = f_refix_range_err
        else:
            saccade_err = f_saccade_err
            saccade_range_err = f_saccade_range_err
    else:  # regression
        if refix:
            saccade_err = r_refix_err
            saccade_range_err = r_refix_range_err
        else:
            saccade_err = regression_err
            saccade_range_err = regression_range_err

    range_error = saccade_err * (saccade_range_err - abs(intended_amplitude))
    range_error_std = saccade_gaussian_err + saccade_random_err * abs(intended_amplitude)
    random_error = np.random.normal(0, range_error_std, 1)
    actual_amplitude = intended_amplitude + range_error + random_error[0]

    post_saccade_k = k + int(actual_amplitude)

    return actual_amplitude, post_saccade_k


def get_saccade_delta():
    return get_random_time(random_timing)[0]


def calculate_activation_integral(y, timestamp, timestep):
    upper_range = int((timestamp - time_delay) / timestep)

    if upper_range <= 0:
        return 0

    activation_integral = calculate_integral(y[:upper_range + 1], dx=timestep)
    return activation_integral / time_delay


def check_for_labile_start(t, prev_saccade, saccade_delta, activation_integral):
    return t > prev_saccade + saccade_delta + inhibition_factor_h * activation_integral


def start_labile_stage():
    return get_random_time(labile_stage)[0]


def start_nonlabile_stage(current_activations, k, word_pos, current_word):
    target_word = saccade_target_selection(current_activations)
    target_pos = find_word_centre(word_pos[target_word, :])
    refix = target_word == current_word

    actual_amplitude, post_saccade_k = calculate_saccade_amplitude(k, target_pos, refix)

    modulated_nonlabile = nonlabile_stage + latency_mod_k0 * np.exp(-latency_mod_k1*actual_amplitude**2)
    timer = get_random_time(modulated_nonlabile)[0]

    return target_word, timer, post_saccade_k


def start_saccade_execution():
    timer = get_random_time(execution_time)

    return timer, suppression_delay
