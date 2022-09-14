def find_fixated_word(k, word_pos):
    assert k <= word_pos[-1, 1]

    for i in range(word_pos.shape[0]):
        if (k >= word_pos[i, 0] - 1) and (k <= word_pos[i, 1]):
            return i
