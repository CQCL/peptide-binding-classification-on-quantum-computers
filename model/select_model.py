from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator

from model.amgen_models import *
from model.utils import *


def get_pure_qrnn(model_args,
                  batch_size=16,
                  device=torch.device('cpu')):

    vocab_size, dataloader, X_train, y_train, X_val, y_val, vocab= \
        get_random_embed_data(device, batch_size=batch_size)

    model = PureQRNN(vocab_size=vocab_size, **model_args).to(device)

    return model, dataloader, X_train, y_train, X_val, y_val


def get_single_embed_qrnn(model_args,
                          batch_size=16,
                          device=torch.device('cpu')):

    dataloader, X_train, y_train, X_val, y_val = \
        get_pre_embed_data(device, batch_size=batch_size)

    model = SingleEmbedQRNN(classical_embed_dim=X_val.shape[2],
                            **model_args).to(device)

    return model, dataloader, X_train, y_train, X_val, y_val


def get_random_embed_qrnn(model_args,
                          batch_size=16,
                          device=torch.device('cpu')):

    vocab_size, dataloader, X_train, y_train, X_val, y_val, vocab = \
        get_random_embed_data(device, batch_size=batch_size)

    model = RandomEmbedQRNN(vocab_size=vocab_size, **model_args).to(device)

    return model, dataloader, X_train, y_train, X_val, y_val


def get_position_embed_qrnn(model_args,
                            batch_size=16,
                            device=torch.device('cpu')):

    dataloader, X_train, y_train, X_val, y_val = \
        get_pre_embed_data(device, batch_size=batch_size)

    model = PositionEmbedQRNN(max_len=X_val.shape[1],
                              classical_embed_dim=X_val.shape[2],
                              **model_args).to(device)

    return model, dataloader, X_train, y_train, X_val, y_val


def get_random_embed_data(device, batch_size=16, shuffle=False):
    peptides, labels = load_data()

    if shuffle:
        for p in peptides:
            np.random.shuffle(p)

    peptides = add_end_token(peptides)
    max_len = max(len(p) for p in peptides)

    for p in peptides:
        p += ['<pad>'] * (max_len - len(p))
        assert len(p) == max_len

    def yield_data(data):
        for p in data:
            yield p

    vocab = build_vocab_from_iterator(yield_data(peptides), specials=['<pad>'])

    peptides = [vocab(p) for p in peptides]
    peptides = torch.tensor(peptides)
    labels = torch.tensor(labels)

    X_train, X_val, y_train, y_val = train_test_split_(peptides, labels)
    X_train = X_train.to(device, dtype=torch.long)
    y_train = y_train.to(device, dtype=torch.long)
    X_val = X_val.to(device, dtype=torch.long)
    y_val = y_val.to(device, dtype=torch.long)

    train_set = TensorDataset(X_train, y_train)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return len(vocab), dataloader, X_train, y_train, X_val, y_val, vocab


def get_pre_embed_data(device, batch_size=16):
    peptides, labels = load_data_tensors()

    X_train, X_val, y_train, y_val = train_test_split_(peptides, labels)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    dataset = TensorDataset(X_train, y_train)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, X_train, y_train, X_val, y_val


def get_pure_qrnn_kfold(model_args,
                        vocab,
                        device=torch.device('cpu')):

    model = PureQRNN(vocab_size=len(vocab), **model_args).to(device)

    return model


def get_random_embed_qrnn_kfold(model_args,
                                vocab,
                                device=torch.device('cpu')):

    model = RandomEmbedQRNN(vocab_size=len(vocab), **model_args).to(device)

    return model


def get_random_embed_qrnn_captum(model_args,
                                 vocab,
                                 device=torch.device('cpu')):

    model = RandomEmbedQRNNCaptum(vocab_size=len(vocab), **model_args).to(device)

    return model


def get_random_embed_data_kfold(device, fold_num, batch_size=16, shuffle=False):
    peptides, labels = load_data(new=True)

    train_indices, val_indices, test_indices = load_kfold_indices()

    if shuffle:
        for p in peptides:
            np.random.shuffle(p)

    peptides = add_end_token(peptides)
    max_len = max(len(p) for p in peptides)

    for p in peptides:
        p += ['<pad>'] * (max_len - len(p))
        assert len(p) == max_len

    def yield_data(data):
        for p in data:
            yield p

    vocab = build_vocab_from_iterator(yield_data(peptides), specials=['<pad>'])

    peptides = [vocab(p) for p in peptides]
    peptides = torch.tensor(peptides)
    labels = torch.tensor(labels)

    X_train = peptides[train_indices[fold_num]]
    y_train = labels[train_indices[fold_num]]

    X_val = peptides[val_indices[fold_num]]
    y_val = labels[val_indices[fold_num]]

    X_test = peptides[test_indices[fold_num]]
    y_test = labels[test_indices[fold_num]]

    X_train = X_train.to(device, dtype=torch.long)
    y_train = y_train.to(device, dtype=torch.long)
    X_val = X_val.to(device, dtype=torch.long)
    y_val = y_val.to(device, dtype=torch.long)
    X_test = X_test.to(device, dtype=torch.long)
    y_test = y_test.to(device, dtype=torch.long)

    train_set = TensorDataset(X_train, y_train)
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return len(vocab), dataloader, X_train, y_train, X_val, y_val, X_test, y_test, vocab