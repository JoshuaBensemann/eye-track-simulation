import string
import pandas as pd
import torch
from wordfreq import word_frequency
from preprocessing.swift_2 import max_activation
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertLMHeadModel


def partition_sentences(df):
    sentence_indices = []
    sentence_start = 0

    for i, row in df.iterrows():
        if row["word"][-1] in [".", "!", "?"]:
            sentence_end = i + 1
            sentence_indices.append([sentence_start, sentence_end])
            sentence_start = sentence_end

    return sentence_indices


def partition_texts(df, id_col="Word_ID"):
    text_indices = []
    text_id = None
    text_start = 0

    for i, row in df.iterrows():
        current_id = row[id_col].split("_")[0]
        if text_id is None:
            text_id = current_id
        else:
            if text_id != current_id:
                text_end = i
                text_indices.append([text_start, text_end])
                text_start = text_end
                text_id = current_id

    return text_indices


def swift_2_process_df(df):
    start = 0
    word = 1

    data_dict = {}

    for i, row in df.iterrows():
        stop = start + row["len"] - 1
        data_dict[i] = {"word_num": word, "start": start, "stop": stop}

        word = word + 1
        start = stop + 2

    word_info = pd.DataFrame.from_dict(data_dict, orient="index")

    df = pd.concat([df, word_info], axis=1)
    df["ln"] = df["freq"].apply(max_activation)
    return df


def get_freq_per_million(word, lang='en', min_value=1):
    min_value = min_value * 1e-6

    return word_frequency(word, lang, minimum=min_value) * 1e6


def words_to_tokens(text, tokenizer):
    words_original = text.split()
    text = text.lower()

    for character in string.punctuation:
        text = text.replace(character, '')

    words = text.split()

    num_words = len(words)

    sentences = []
    for i in range(num_words):
        sentences.append(" ".join(words[0:i + 1]))

    tokens = tokenizer.batch_encode_plus(sentences, padding=True, return_tensors="np")["input_ids"]
    token_tensor = tokenizer.encode(text, return_tensors="pt")

    return words_original, tokens, token_tensor


def word_probs_to_dataframe(word_probs, text_id):
    df = pd.DataFrame.from_dict(word_probs, orient="index").reset_index()
    df["Word_ID"] = df["index"].apply(lambda s: f"{text_id}_{s.split('_')[0]}")
    df["word"] = df["index"].apply(lambda s: s.split("_")[1])
    df["len"] = df["word"].apply(len)
    df = df.rename({0: "prob"}, axis=1)
    df["freq"] = df["word"].apply(get_freq_per_million)

    return df[["Word_ID", "word", "len", "prob", "freq"]]


class DataFrameCreator():

    def __init__(self, base_model):
        self.base_model = base_model

        if "gpt2" in self.base_model:
            self.model = GPT2LMHeadModel.from_pretrained(base_model)
            self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.special_tokens = self.tokenizer.all_special_ids

        if "bert" in self.base_model:
            self.tokenizer = BertTokenizer.from_pretrained(base_model)
            self.model = BertLMHeadModel.from_pretrained(base_model, is_decoder=True)
            self.special_tokens = [102]

        assert self.model is not None

    def find_special_tokens(self, tokens):
        for i in range(tokens.shape[0]):
            if tokens[i] in self.special_tokens:
                return i

        return None

    def get_probabilities_from_model(self, words, tokens, tokens_tensor):
        logits = self.model(tokens_tensor)["logits"][0]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        word_probs = {f"1_{words[0]}": 0}
        next_word_start = self.find_special_tokens(tokens[0])

        for i in range(1, tokens.shape[0]):
            prob_list = []
            word_start = next_word_start
            next_word_start = self.find_special_tokens(tokens[i])
            if next_word_start is None:
                next_word_start = word_start + 1

            for j in range(word_start, next_word_start):
                prob = probs[j - 1, tokens_tensor[0, j]].item()
                prob_list.append(prob)

            word_probs[f"{i+1}_{words[i]}"] = min(prob_list)

        return word_probs

    def create_dataframe_from_texts(self, texts):
        dfs = []

        for i, text in enumerate(texts):
            print(f"Converting Text {i+1} of {len(texts)}!")
            try:
                words, tokens, tokens_tensor = words_to_tokens(text, self.tokenizer)
                word_probs = self.get_probabilities_from_model(words, tokens, tokens_tensor)
                dfs.append(word_probs_to_dataframe(word_probs, i+1))
            except:
                pass

        return pd.concat(dfs)


