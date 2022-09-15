import numpy as np

freq_intercept = 63.5
freq_slop = -0.2
predictability = 0.11
F = 11


def max_activation(freq):  # Equation 1
    ln = freq_intercept * (1 + freq_slop * (np.log(freq) / F))
    return ln


def process_sentence(df):
    num_words = df.shape[0]
    difficulty = np.zeros(num_words)
    word_complete = np.zeros(num_words, dtype="int")
    word_pos = np.zeros([num_words, 2], dtype="int")
    word_prob = np.zeros(num_words)

    for i in range(num_words):
        difficulty[i] = df["ln"].iloc[i]
        word_pos[i, 0] = df["start"].iloc[i]
        word_pos[i, 1] = df["stop"].iloc[i]
        word_prob[i] = df["prob"].iloc[i]

    return word_complete, difficulty, word_pos, word_prob
