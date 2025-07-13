import torch
import torch.nn as nn
from typing import List
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


class Config:
    exeriment_name = (
        "skills2vec" + "_" + str(pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    min_count = 5
    sg = 1  # sg is 1, then skip-gram otherwise CBOW
    window = 3
    vector_size = 300
    padding = "<pad>"
    is_padding = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = 10
    lr = 0.001
    batch_size = 64
    sample = False
    model_type = "cbow"  # or 'skip-gram'


class CustomW2vModel(nn.Module):
    config: Config

    def most_similar(self, word, topn=10):
        # This method should return the top n most similar words to the given word
        if word not in self.word2idx:
            raise ValueError(f"Word '{word}' not found in vocabulary.")
        word_idx = self.word2idx[word]
        word_vector = self.project_layer(
            torch.tensor(word_idx).to(self.project_layer.weight.device)
        )
        word_vector = word_vector.unsqueeze(0)  # Add batch dimension
        similarities = torch.matmul(self.project_layer.weight, word_vector.T).squeeze(1)

    def n_smilarity(self, words1, word2):
        pass

    def save(self, path):
        # This method should save the model to the specified path
        torch.save(self.state_dict(), path)


class SkipGramModel(CustomW2vModel):
    def __init__(self, config: Config, vocab_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_size = config.vector_size
        self.vocab_size = vocab_size
        self.window = config.window

        self.project_layer = nn.Embedding(self.vocab_size, self.vector_size)
        self.linar = nn.Linear(self.vector_size, self.vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x_embed = self.project_layer(x)
        out = self.linar(x_embed)
        out = self.softmax(out)
        return out


# add -100 for padding


class CBOWModel(CustomW2vModel):
    def __init__(self, config: Config, vocab_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_size = config.vector_size
        self.vocab_size = vocab_size

        self.project_layer = nn.Embedding(self.vocab_size, self.vector_size)
        self.linar = nn.Linear(self.vector_size, self.vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # This method should implement the forward pass of the CBOW model
        context = self.project_layer(x)
        avg_embed = context.mean(dim=1)
        out = self.linar(avg_embed)
        out = self.softmax(out)
        return out
