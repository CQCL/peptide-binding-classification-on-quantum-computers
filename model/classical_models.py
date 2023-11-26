import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


model_type_map = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}


class Embedding_RNN(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.rnn = model_type_map[model_type](input_size, hidden_size,
                                              num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, x_lens):
        x = pack_padded_sequence(x, x_lens, batch_first=True,
                                 enforce_sorted=False)
        out, _ = self.rnn(x)
        out, lens = pad_packed_sequence(out, batch_first=True)
        out = self.linear(out[torch.arange(out.shape[0]), lens-1])

        return out


class Random_RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, model_type, hidden_size,
                 num_layers=1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=0)
        self.rnn = model_type_map[model_type](embed_dim, hidden_size,
                                              num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x, x_lens):
        x = self.embedding(x)
        x = pack_padded_sequence(x, x_lens, batch_first=True,
                                 enforce_sorted=False)
        out, _ = self.rnn(x)
        out, lens = pad_packed_sequence(out, batch_first=True)
        out = self.linear(out[torch.arange(out.shape[0]), lens-1])
        out = torch.sigmoid(out)

        return out


class RandomRNNCaptum(Random_RNN):
    def __init__(self, vocab_size, embed_dim, model_type, hidden_size,
                 num_layers=1):
        super().__init__(vocab_size, embed_dim, model_type, hidden_size,
                         num_layers)

    def get_embeddings(self, x):
        return self.embedding(x)

    def forward(self, x_embs, x_lens):
        x = pack_padded_sequence(x_embs, x_lens, batch_first=True,
                                 enforce_sorted=False)
        out, _ = self.rnn(x)
        out, lens = pad_packed_sequence(out, batch_first=True)
        out = self.linear(out[torch.arange(out.shape[0]), lens-1])
        out = torch.sigmoid(out)

        return out
