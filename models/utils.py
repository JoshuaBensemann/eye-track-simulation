import numpy as np
from scipy.integrate import trapezoid


def find_fixated_word(k, word_pos):
    if k < 0:
        return -1

    if k >= word_pos[-1, 1]:
        return word_pos.shape[0]

    for i in range(word_pos.shape[0]):
        if (k >= word_pos[i, 0] - 1) and (k <= word_pos[i, 1]):
            return i


def find_word_centre(word_pos):
    return int((word_pos[0] + word_pos[1]) / 2)


def get_random_time(ave, samples=1):
    return np.random.gamma(ave, size=samples)


def calculate_integral(y, dx=1):
    return trapezoid(y, dx=dx)
