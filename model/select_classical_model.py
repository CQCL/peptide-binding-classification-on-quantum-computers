from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from model.classical_models import *
from model.utils import *


def get_pad_collate(device):
    def pad_collate(batch):
        (xx, y) = zip(*batch)
        x_lens = [len(x) for x in xx]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).to(device)

        return xx_pad, x_lens, torch.tensor(y).to(device)

    return pad_collate


def get_pretrained_classical_rnn(model_args,
                                 batch_size=16,
                                 use_gpu=torch.cuda.is_available()):
    if use_gpu:
        device = get_idle_gpu()
    else:
        device = torch.device('cpu')

    peptides, labels = load_data_tensors_unshaped()
    labels = torch.tensor(labels)

    train_loader, train_eval_loader, val_loader = \
        get_dataloaders(peptides, labels, device, batch_size)

    model = Embedding_RNN(**model_args).to(device)

    return model, train_loader, train_eval_loader, val_loader


def get_random_classical_rnn(model_args,
                             batch_size=16,
                             use_gpu=torch.cuda.is_available()):
    if use_gpu:
        device = get_idle_gpu()
    else:
        device = torch.device('cpu')

    peptides, labels = load_data()
    peptides = add_end_token(peptides)

    def yield_data(data):
        for p in data:
            yield p

    vocab = build_vocab_from_iterator(yield_data(peptides))

    peptides = [torch.tensor(vocab(p)) for p in peptides]
    labels = torch.tensor(labels)

    train_loader, train_eval_loader, val_loader = \
        get_dataloaders(peptides, labels, device, batch_size)

    model = Random_RNN(vocab_size=len(vocab), **model_args).to(device)

    return model, train_loader, train_eval_loader, val_loader


def get_random_classical_rnn_kfold(model_args,
                                   datasets,
                                   vocab,
                                   batch_size=16,
                                   device=torch.device('cpu')):
    X_train, y_train = datasets[0].tensors
    X_val, y_val = datasets[1].tensors
    X_test, y_test = datasets[2].tensors

    X_train = padded_tensor_to_list(X_train)
    X_val = padded_tensor_to_list(X_val)
    X_test = padded_tensor_to_list(X_test)

    train_set = Text_Dataset(X_train, y_train)
    val_set = Text_Dataset(X_val, y_val)
    test_set = Text_Dataset(X_test, y_test)

    pad_collate = get_pad_collate(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate)
    train_eval_loader = DataLoader(train_set, batch_size=len(train_set),
                                   shuffle=False, collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False,
                            collate_fn=pad_collate)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False,
                             collate_fn=pad_collate)

    model = Random_RNN(vocab_size=len(vocab), **model_args).to(device)

    return model, train_loader, train_eval_loader, val_loader, test_loader


def get_random_classical_rnn_captum(model_args,
                                    datasets,
                                    vocab,
                                    batch_size=16,
                                    device=torch.device('cpu')):
    X_train, y_train = datasets[0].tensors
    X_val, y_val = datasets[1].tensors
    X_test, y_test = datasets[2].tensors

    X_train = padded_tensor_to_list(X_train)
    X_val = padded_tensor_to_list(X_val)
    X_test = padded_tensor_to_list(X_test)

    train_set = Text_Dataset(X_train, y_train)
    val_set = Text_Dataset(X_val, y_val)
    test_set = Text_Dataset(X_test, y_test)

    pad_collate = get_pad_collate(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate)
    train_eval_loader = DataLoader(train_set, batch_size=len(train_set),
                                   shuffle=False, collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False,
                            collate_fn=pad_collate)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False,
                             collate_fn=pad_collate)

    model = RandomRNNCaptum(vocab_size=len(vocab), **model_args).to(device)

    return model, train_loader, train_eval_loader, val_loader, test_loader


def get_dataloaders(peptides, labels, device, batch_size=16):
    X_train, X_val, y_train, y_val = train_test_split_(peptides, labels)
    train_set = Text_Dataset(X_train, y_train)
    val_set = Text_Dataset(X_val, y_val)

    pad_collate = get_pad_collate(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=pad_collate)
    train_eval_loader = DataLoader(train_set, batch_size=len(train_set),
                                   shuffle=False, collate_fn=pad_collate)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False,
                            collate_fn=pad_collate)

    return train_loader, train_eval_loader, val_loader


def padded_tensor_to_list(tensor):
    return [t[:torch.argmin(t)] for t in tensor]