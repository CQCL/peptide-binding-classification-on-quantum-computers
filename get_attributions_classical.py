import copy
import random
import sys

import captum
from captum.attr import IntegratedGradients, GradientShap, DeepLift, FeatureAblation, ShapleyValueSampling
import matplotlib.pyplot as plt
import pickle
import torch
import os

from model.ansatz import *
from model.select_classical_model import get_random_classical_rnn_captum
from model.utils import *


arg_type_map = {'embed_type': str, 'model_type': str, 'hidden_size': int,
                'num_layers': int, 'embed_dim': int,
                'learning_rate': float, 'batch_size': int,
                'epochs': int, 'weight_decay': float, 'seed': int,
                'use_gpu': str2bool, 'log_dir': str}

model_args = {'model_type': str, 'hidden_size': int, 'num_layers': int}
required = {'embed_type', 'model_type', 'hidden_size', 'log_dir'}
defaults = {'num_layers': 1, 'learning_rate': 1e-3, 'batch_size': 16,
            'epochs': 100, 'weight_decay': 0.0,
            'use_gpu': torch.cuda.is_available()}


def ig_attributions(model, X, X_lens):
    ig = IntegratedGradients(model)

    attributions, delta = ig.attribute(X,
                                       baselines=0,
                                       n_steps=50,
                                       additional_forward_args=(X_lens),
                                       return_convergence_delta=True)

    attributions = attributions.cpu().detach().numpy()
    delta = delta.cpu().detach().numpy()

    return attributions, delta


def dl_attributions(model, X, X_lens):
    dl = DeepLift(model)

    # DeepLift requires an even number of samples
    if(X.shape[0] % 2 == 1):
        # append a zero vector to X
        with torch.no_grad():
            X_emb = torch.cat((X, torch.zeros_like(X[0]).unsqueeze(0)), dim=0)
            X_emb_lens = X_lens + [1]
    else:
        X_emb = X
        X_emb_lens = X_lens

    attributions, delta = dl.attribute(X_emb,
                                       baselines=0,
                                       additional_forward_args=(X_emb_lens),
                                       return_convergence_delta=True)

    if X.shape[0] % 2 == 1:
        attributions = attributions[:-1]
        delta = delta[:-1]

    attributions = attributions.cpu().detach().numpy()
    delta = delta.cpu().detach().numpy()

    return attributions, delta


def gs_attributions(model, X, X_lens):
    gs = GradientShap(model)

    attributions = []
    delta = []
    for i in range(X.shape[0]):
        i_attributions, i_delta = gs.attribute(X[i].unsqueeze(0),
                                               baselines=X,
                                               stdevs=0.1,
                                               n_samples=100,
                                               additional_forward_args=([X_lens[i]]),
                                               return_convergence_delta=True)
        attributions.append(i_attributions)
        delta.append(i_delta)

    attributions = torch.cat(attributions, dim=0)
    delta = torch.cat(delta, dim=0)

    attributions = attributions.cpu().detach().numpy()
    delta = delta.cpu().detach().numpy()

    return attributions, delta


def fa_attributions(model, X, X_lens):
    fa = FeatureAblation(model)

    mask = torch.tensor([[i] * X.shape[2] for i in range(X.shape[1])])
    mask = mask.unsqueeze(0).repeat(X.shape[0], 1, 1)

    attributions = fa.attribute(X,
                                baselines=0,
                                additional_forward_args=(X_lens),
                                feature_mask=mask)

    attributions = attributions.cpu().detach().numpy()
    delta = torch.zeros_like(X).cpu().detach().numpy()

    return attributions, delta


def svs_attributions(model, X, X_lens):
    svs = ShapleyValueSampling(model)

    mask = torch.tensor([[i] * X.shape[2] for i in range(X.shape[1])])
    mask = mask.unsqueeze(0).repeat(X.shape[0], 1, 1)

    attributions = svs.attribute(X,
                                 baselines=0,
                                 additional_forward_args=(X_lens),
                                 feature_mask=mask)

    attributions = attributions.cpu().detach().numpy()
    delta = torch.zeros_like(X).cpu().detach().numpy()

    return attributions, delta


interpretation_methods = {'ig': ig_attributions, 'gs': gs_attributions,
                          'dl': dl_attributions, 'fa': fa_attributions,
                          'svs': svs_attributions}


def eval(model, dataloader):
    model.eval()

    for X, X_lens, y in dataloader:
        X = model.get_embeddings(X)
        acc = torch_binary_acc(model(X, X_lens), y)
        f1 = torch_f1(model(X, X_lens), y)
        break

    return acc, f1


def all_attributions(model, test_loader, i_method_key):
    for X, X_lens, y in test_loader:
        break

    with torch.no_grad():
        X_embs = model.get_embeddings(X)
        probs = model(X_embs, X_lens)

    attributions_pos = []
    attributions_neg = []
    deltas_pos = []
    deltas_neg = []
    pred_class_pos = []
    pred_class_neg = []
    pos_sentences = []
    neg_sentences = []

    att, delta = interpretation_methods[i_method_key](model, X_embs, X_lens)

    for i in range(len(X)):
        pred_class = int(probs[i].item() >= 0.5)
        if y[i].item() == 1:
            attributions_pos.append(att[i])
            deltas_pos.append(delta[i])
            pred_class_pos.append(pred_class)
            pos_sentences.append(X[i].detach().cpu().numpy())
        else:
            attributions_neg.append(att[i])
            deltas_neg.append(delta[i])
            pred_class_neg.append(pred_class)
            neg_sentences.append(X[i].detach().cpu().numpy())

    return (np.array(attributions_pos), np.array(attributions_neg), np.array(pred_class_pos), np.array(pred_class_neg),
            np.array(pos_sentences), np.array(neg_sentences), np.array(deltas_pos), np.array(deltas_neg))


def draw_attributions(attributions_pos, attributions_neg, pred_class_pos, pred_class_neg, pos_sentences, neg_sentences, vocab, filename):
    # fig, axs = plt.subplots(1, 2, gridspec_kw={'height_ratios': [1, ratio]})
    fig = plt.figure(figsize=(6,15))
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[1, 1], height_ratios=[1, (len(attributions_neg)-len(attributions_pos)) / len(attributions_pos)])

    attributions_pos = np.sum(attributions_pos, axis=2)
    attributions_neg = np.sum(attributions_neg, axis=2)
    # attributions_pos = attributions_pos / np.sum(np.abs(attributions_pos), axis=1, keepdims=True)
    # attributions_neg = attributions_neg / np.sum(np.abs(attributions_neg), axis=1, keepdims=True)
    attributions_pos = attributions_pos / np.linalg.norm(attributions_pos, axis=1, keepdims=True)
    attributions_neg = attributions_neg / np.linalg.norm(attributions_neg, axis=1, keepdims=True)

    # Actually positive
    ax1 = fig.add_subplot(spec[0, 0])
    p = ax1.imshow(attributions_pos, cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)

    # Positive but predicted negative
    incorrect_pos = np.where(np.array(pred_class_pos) != 1)[0]
    for i in incorrect_pos:
        ax1.text(attributions_pos.shape[1], i, 'x', color='red', fontsize=12, fontweight='bold', va='center')
        ax1.axhline(i, color='red', linewidth=0.5)


    # Actually negative
    ax2 = fig.add_subplot(spec[0:, 1])
    n = ax2.imshow(attributions_neg, cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)

    # Negative but predicted positive
    incorrect_neg = np.where(np.array(pred_class_neg) != 0)[0]
    for i in incorrect_neg:
        ax2.text(attributions_neg.shape[1], i, 'x', color='red', fontsize=12, fontweight='bold', va='center')
        ax2.axhline(i, color='red', linewidth=0.5)

    ax1.set_title("Positive sentences")
    ax2.set_title("Negative sentences")

    for ax in [ax1, ax2]:
        ax.set_anchor('NW')
        ax.set_aspect('equal')
        # ax.set_xlabel("Character in sequence")
        #Hide x ticks and labels
        ax.set_xticks([])
        ax.set_xticklabels([])

    # cax = fig.add_subplot(spec[0:, 2])
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.tight_layout()
    cax = fig.add_axes([1.05, 0.3, 0.02, 0.6], anchor='NE')

    cb = fig.colorbar(p, cax=cax)
    cb.set_label('Attribution value', rotation=270, labelpad=15)

    # Display text for each sentence
    for ax, sentences in zip([ax1, ax2], [pos_sentences, neg_sentences]):
        for i, sentence in enumerate(sentences):
            sentence = vocab.lookup_tokens(sentence)
            for j, char in enumerate(sentence):
                if not (char == '<pad>' or char == 'END'):
                    ax.text(j, i, char, fontsize=8, va='center', ha='center')


    ax1.set_ylabel("Sentence number")

    num_pos_sentences = len(pos_sentences)
    num_neg_sentences = len(neg_sentences)

    plt.text(0.5, 1.08, f"Positive sentences: {num_pos_sentences}", ha='center', va='center', transform=ax1.transAxes)
    plt.text(0.5, 1.05, f"Negative sentences: {num_neg_sentences}", ha='center', va='center', transform=ax2.transAxes)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

i_method_key = sys.argv[1]
if i_method_key not in interpretation_methods:
    raise ValueError(f'Invalid interpretation method: {i_method_key}')

filepath = sys.argv[2]
with open(filepath, 'r') as f:
    lines = f.readlines()
    args = lines[0].split(',')
    args = [a.strip() for a in args]
    c = lines[1].split(',')
    config = [a.strip() for a in c]

shared_config = dict(zip(args, [arg_type_map[arg](config[i]) for i, arg in enumerate(args)]))
vocab, fold_datasets = load_dataset_folds()


shared_config = {**defaults, **shared_config}

log_dir = shared_config['log_dir']
if not os.path.exists(log_dir):
    raise ValueError(f'Log directory does not exist: {log_dir}')

results_loc = log_dir.split('/')[1]
results_loc = f'{results_loc}_{i_method_key}'

config_name = log_dir.split('/')[1]
print(f'Running config: {config_name}')

results_dir = os.path.join('captum_results', results_loc)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if shared_config['use_gpu']:
    device = get_idle_gpu()
else:
    device = torch.device('cpu')

model_results = {}
model_accuracies = {}
model_f1s = {}
model_attributions = {}

max_fold_accs = []
max_fold_f1s = []

for fold, datasets in enumerate(fold_datasets):
    print(f'Running fold: {fold}')

    fold_dir = os.path.join(log_dir, f'fold_{fold}')
    if not os.path.exists(fold_dir):
        continue
    results_fold_dir = os.path.join(results_dir, f'fold_{fold}')
    if not os.path.exists(results_fold_dir):
        os.makedirs(results_fold_dir)

    fold_accuracies = {}
    fold_f1s = {}
    fold_attributions = {}

    max_fold_f1 = 0
    max_fold_f1_acc = 0

    for seed in [0, 2, 4]:
        print(f'Running with seed: {seed}')

        config = copy.copy(shared_config)
        iter_dir = os.path.join(fold_dir, f'iter_{seed}')
        if not os.path.exists(iter_dir):
            continue

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("Loading data and initialising model...")

        model_args = {arg: config[arg] for arg in model_args}
        if config['embed_type'] == 'random':
            model_args['embed_dim'] = config['embed_dim']

        model, train_loader, train_eval_loader, val_loader, test_loader = \
            get_random_classical_rnn_captum(model_args,
                                            datasets,
                                            vocab,
                                            batch_size=config['batch_size'],
                                            device=device)
        config['num_params'] = sum(p.numel() for p in model.parameters())

        state_dict = torch.load(os.path.join(iter_dir, 'best_model.pt'), map_location=device)
        model.load_state_dict(state_dict)

        model.eval()
        test_acc, test_f1 = eval(model, test_loader)
        print(test_acc, test_f1)
        fold_accuracies[seed] = test_acc
        fold_f1s[seed] = test_f1

        if test_f1 > max_fold_f1:
            max_fold_f1 = test_f1
            max_fold_f1_acc = test_acc

        model.train()

        attributions_pos, attributions_neg, predicted_pos, predicted_neg, pos_sentences, neg_sentences, deltas_pos, deltas_neg = \
            all_attributions(model, test_loader, i_method_key)
        draw_attributions(attributions_pos, attributions_neg, predicted_pos, predicted_neg, pos_sentences,
                            neg_sentences, vocab, os.path.join(results_fold_dir, f'iter_{seed}.png'))

        fold_attributions[seed] = {'attributions_pos': attributions_pos,
                                   'attributions_neg': attributions_neg,
                                   'pred_class_pos': predicted_pos,
                                   'pred_class_neg': predicted_neg,
                                   'sentences_pos': pos_sentences,
                                   'sentences_neg': neg_sentences,
                                   'deltas_pos': deltas_pos,
                                   'deltas_neg': deltas_neg,}


    model_accuracies[fold] = fold_accuracies
    model_f1s[fold] = fold_f1s
    model_attributions[fold] = fold_attributions
    max_fold_accs.append(max_fold_f1_acc)
    max_fold_f1s.append(max_fold_f1)

model_results['accuracies'] = model_accuracies
model_results['f1s'] = model_f1s
model_results['attributions'] = model_attributions
model_results['max_fold_accs'] = max_fold_accs
model_results['max_fold_f1s'] = max_fold_f1s
model_results['mean_acc'] = np.mean(max_fold_accs)
model_results['std_acc'] = np.std(max_fold_accs)
model_results['mean_f1'] = np.mean(max_fold_f1s)
model_results['std_f1'] = np.std(max_fold_f1s)
model_results['config'] = config

with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(model_results, f)
