import torch
from typing import List
from collections import defaultdict
from model import *
import pandas as pd


def calculate_word2idx(words: List[List[str]], min_count=1):
    word_count = defaultdict(int)
    for word in words:
        for w in word:
            if len(w) > 0:
                word_count[w] += 1
    # print(word_count)
    idx = 0
    w2i = {}
    for _, (w, val) in enumerate(word_count.items()):
        if val >= min_count:
            w2i[w] = idx
            idx += 1

    if Config.is_padding:
        w2i[Config.padding] = len(w2i) + 1  # Add padding token
    assert len(w2i) > 0
    return w2i



def prepare_training_data(words, word2idx, window, sg = 1):
    # This function should prepare the training data for the CBOW model
    training_data = []
    for sent in words:
        indices = [word2idx[word] for word in sent if word in word2idx]
        for i, idx in enumerate(indices):
            start = max(0, i - window)
            end = min(len(indices), i + window + 1)
            context = [indices[j] for j in range(start, end) if i != j]
            if sg == 1:
                training_data.extend([[[w],idx] for w in context])
            else:
                if len(context) == window * 2:
                    training_data.append([context, idx])
                elif len(context) < window * 2 and Config.is_padding:
                    # If context is smaller than expected, we can pad it with zeros
                    context = context + [0] * (window * 2 - len(context))
                    training_data.append([context, idx])

    return training_data



class word2vecDataset(torch.utils.data.Dataset):
    def __init__(self, words: List[List[str]], config: Config, sg:int = 1):
        self.word2idx = calculate_word2idx(words, config.min_count)
        self.training_data = prepare_training_data(words, self.word2idx, config.window, sg)
        self.vector_size = config.vector_size
        self.vocab_size = len(self.word2idx)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        context, target = self.training_data[idx]
        return torch.tensor(context), torch.tensor(target)


def load_dataset(path: str):
    df = pd.read_pickle(
        path
    ) 
    words = df.values.tolist()
    return words
