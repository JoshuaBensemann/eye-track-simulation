import numpy as np

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


def calculate_preprocessing_factor(word_complete, word_pos, word_prob):  # Equation 8
    num_words = word_complete.shape[0]
    preprocess_factors = np.zeros(num_words)

    for i in range(num_words):
        prob = word_prob[i]

        if word_complete[i] == 1:  # Lexical completion
            preprocess_factors[i] = -(1 + prob)

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
                      time_step=2):  # Equation 7
    preprocess_factors = calculate_preprocessing_factor(word_complete, word_pos, word_prob)
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
Saccard Programming
"""


def saccard_target_selection(current_processed):
    return current_processed / np.sum(current_processed)


def check_for_saccard(t, prev_saccard, delta, activation):
    return t > prev_saccard + delta + inhibition_factor_h * activation
