import os
import subprocess as sp

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import torch
from torch.utils.data import Dataset
import pickle

eps = 1e-8


def load_dataset_folds():
    with open("./Additional Data/split_data.pkl", "rb") as f:
        data = pickle.load(f)

    return data["vocab"], data["datasets"]


def load_dataset_folds_no_end_token():
    with open('./Additional Data/split_data_no_end_token.pkl', 'rb') as f:
        data = pickle.load(f)

    return data['vocab'], data['datasets']


def load_dataset_folds_bigrams():
    with open('./Additional Data/split_data_bigrams.pkl', 'rb') as f:
        data = pickle.load(f)

    return data['vocab'], data['datasets']


def load_data(new=False, return_ids=False):
    peptides, labels, ids = [], [], []

    if new:
        loc = "Additional Data/Sequences and Labels new 7030.csv"
    else:
        loc = "./PilotData/Sequences and Labels.csv"

    with open(loc) as f:
        f.readline()
        for line in f:
            id, peptide, label = line.strip().split(",")
            peptides.append(peptide)
            labels.append(label)
            ids.append(id)

    peptides = [[*p] for p in peptides]
    labels = [0 if label == "Weak" else 1 for label in labels]

    if return_ids:
        return peptides, labels, ids

    return peptides, labels


def load_data_tensors():
    _, labels = load_data()

    peptides = []
    for i in range(len(labels)):
        path = f"./PilotData/tensors/Seq{i}.pt"
        peptides.append(torch.load(path))

    max_len = max(p.shape[1] for p in peptides)

    peptides_reshaped = []
    for p in peptides:
        peptides_reshaped.append(
            torch.nn.functional.pad(p, (0, 0, 0, max_len - p.shape[1]))
        )

    peptides = torch.cat(peptides_reshaped, dim=0)
    labels = torch.tensor(labels)

    return peptides, labels.long()


def load_data_tensors_unshaped():
    _, labels = load_data()

    peptides = []
    for i in range(len(labels)):
        path = f"./PilotData/tensors/Seq{i}.pt"
        peptides.append(torch.load(path).squeeze(0))

    return peptides, labels


def add_end_token(peptides):
    return [p + ["END"] for p in peptides]


def acid_counts_prop(peptides, labels):
    acids = set(sum(peptides, []))
    acid_counts = {acid: (0, 0) for acid in acids}

    for a in acids:
        for i, p in enumerate(peptides):
            if a in p and labels[i] == 0:
                acid_counts[a] = (acid_counts[a][0] + 1, acid_counts[a][1])
            elif a in p and labels[i] == 1:
                acid_counts[a] = (acid_counts[a][0], acid_counts[a][1] + 1)

    weak = sum([label == 0 for label in labels])
    strong = len(labels) - weak

    acid_counts_prop = {
        acid: (
            round(acid_counts[acid][0] / weak, 2),
            round(acid_counts[acid][1] / strong, 2),
        )
        for acid in acids
    }

    return acid_counts_prop


def torch_multiclass_acc(predictions, labels):
    with torch.no_grad():
        return torch.sum(torch.argmax(predictions, axis=1) == labels).item() / len(
            predictions
        )


def torch_binary_logits_acc(logits, labels):
    with torch.no_grad():
        preds = logits >= 0
        return (preds.long().squeeze() == labels.squeeze()).sum().item() / len(labels)


def torch_binary_acc(probs, labels):
    with torch.no_grad():
        preds = probs >= 0.5
        return (preds.long().squeeze() == labels.squeeze()).sum().item() / len(labels)


def torch_f1(probs, labels, return_confusion_mat=False):
    with torch.no_grad():
        return f1_score(labels.cpu(), probs.cpu() >= 0.5, average="binary")


def torch_logits_f1(logits, labels, return_confusion_mat=False):
    # with torch.no_grad():
    #     preds = logits >= 0
    #     preds = preds.long().squeeze()
    #     labels = labels.squeeze()

    #     confusion_vector = preds/labels

    #     # Taking 'strong' as the positive class
    #     true_pos = (confusion_vector == 1).sum().item()
    #     false_pos = (confusion_vector == float('inf')).sum().item()
    #     true_neg = (confusion_vector.isnan()).sum().item()
    #     false_neg = (confusion_vector == 0).sum().item()

    #     if return_confusion_mat:
    #         return 2 * true_pos / (2 * true_pos + false_pos + false_neg), (true_pos, true_neg, false_pos, false_neg)

    #     return 2 * true_pos / (2 * true_pos + false_pos + false_neg)
    return f1_score(labels.cpu(), logits.cpu() >= 0, average="binary")


def torch_confusion(probs, labels):
    with torch.no_grad():
        return confusion_matrix(labels.cpu(), probs.cpu() >= 0.5)


def torch_logits_confusion(logits, labels):
    with torch.no_grad():
        return confusion_matrix(labels.cpu(), logits.cpu() >= 0)


def torch_loss(predictions, labels):
    predictions = torch.log(predictions + eps)
    return torch.nn.functional.nll_loss(predictions, labels.long())


def np_acc(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)


def np_loss(predictions, labels):
    labels = [[1, 0] if l == 0 else [0, 1] for l in labels]
    return np.mean(-np.sum(np.log(predictions + eps) * labels, axis=1))


def train_val_test_split(data, labels):
    X_train, X_vt, y_train, y_vt = train_test_split(
        data, labels, stratify=labels, test_size=0.25
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_vt, y_vt, stratify=y_vt, test_size=0.5
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_test_split_(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, stratify=labels, test_size=0.2
    )

    return X_train, X_test, y_train, y_test


def get_collate_fn(vocab, device):
    def collate_fn(batch):
        pep, tar = zip(*batch)
        pep = [vocab(p) for p in pep]
        pep = torch.tensor(pep, dtype=torch.int64).to(device)
        tar = torch.tensor(tar, dtype=torch.int64).to(device)
        return pep, tar

    return collate_fn


def get_idle_gpu():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_available = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    idx = np.argmax(memory_available)

    return torch.device("cuda:" + str(idx))


def str2bool(v):
    return v.lower() not in ("no", "false", "f", "0")


class Text_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_kfold_indices():
    with open("./Additional Data/indices.pickle", "rb") as f:
        data = pickle.load(f)

    train_indices = data["train_indices"]
    val_indices = data["val_indices"]
    test_indices = data["test_indices"]

    return train_indices, val_indices, test_indices


def get_configs(fpath):

    with open(fpath, "r") as f:
        lines = f.readlines()
        args = lines[0].split(",")
        args = [a.strip() for a in args]
        configs = []
        for i in range(1, len(lines)):
            c = lines[i].split(",")
            c = [a.strip() for a in c]
            configs.append(c)

    configs = [
        dict(zip(args, [arg_type_map[arg](config[i]) for i, arg in enumerate(args)]))
        for config in configs
    ]

    return configs


def get_completed_iters(log_dir):

    completed_folds = []
    for root, dirs, files in os.walk(log_dir):
        for dir in dirs:
            if "fold" in dir:
                completed_folds.append(dir)

    completed_iters = []
    for fold in completed_folds:
        fold_iters = []
        fold_dir = os.path.join(log_dir, fold)
        for root, dirs, files in os.walk(fold_dir):
            for dir in dirs:
                if "iter" in dir:
                    fold_iters.append(dir)
        completed_iters.append(fold_iters)

    return completed_folds, completed_iters


def get_best_val_iters():
    with open("tket_best_iters_val.pkl", "rb") as f:
        best_iters_dict = pickle.load(f)
    return best_iters_dict


arg_type_map = {
    "embed_type": str,
    "embed_dim": int,
    "encoder_ansatz": str,
    "encoder_layers": int,
    "recurrent_ansatz": str,
    "recurrent_layers": int,
    "reupload_count": int,
    "n_wires": int,
    "measure": str,
    "learning_rate": float,
    "batch_size": int,
    "epochs": int,
    "weight_decay": float,
    "dropout_prob": float,
    "seed": int,
    "use_gpu": str2bool,
    "log_dir": str,
    "measure": str,
    "final": str,
}
