import numpy as np
import torch
import sympy
import debugpy

import captum
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from typing import List

from model.ansatz import *
from model.select_model import *

arg_type_map = {'embed_type': str, 'embed_dim': int, 'encoder_ansatz': str,
                'encoder_layers': int, 'recurrent_ansatz': str,
                'recurrent_layers': int, 'reupload_count': int,
                'n_wires': int, 'measure': str, 'learning_rate': float,
                'batch_size': int, 'epochs': int, 'weight_decay': float,
                'dropout_prob': float, 'seed': int, 'use_gpu': str2bool,
                'log_dir': str, 'measure': str, 'final': str}

ansatz_map = {'None': None, 'sim7': sim7, 'sim8': sim8, 'sim9': sim9,
              'sim11': sim11, 'sim12': sim12, 'sim13': sim13, 'sim14': sim14,
              'sim17': sim17}

model_args = {'encoder_ansatz', 'encoder_layers', 'recurrent_ansatz',
              'recurrent_layers', 'reupload_count', 'n_wires', 'measure',
              'dropout_prob', 'final'}

embed_model_map = {'pure': get_pure_qrnn_kfold, 'single': get_single_embed_qrnn,
                   'random': get_random_embed_qrnn_kfold,
                   'position': get_position_embed_qrnn}

required = {'embed_type', 'encoder_ansatz', 'n_wires', 'log_dir'}
defaults = {'embed_dim': 0, 'encoder_layers': 1, 'recurrent_ansatz': 'None',
            'recurrent_layers': 1, 'reupload_count': 1, 'measure': 'all',
            'learning_rate': 1e-2, 'batch_size': 16, 'epochs': 100,
            'weight_decay': 0.0, 'dropout_prob': 0.0, 'final': 'linear',
            'use_gpu': False}

def reset_model_args():
     model_args = {'encoder_ansatz', 'encoder_layers', 'recurrent_ansatz',
               'recurrent_layers', 'reupload_count', 'n_wires', 'measure',
               'dropout_prob', 'final'}
     return model_args

device = torch.device('cpu')

def interpret_sentence_for_vis(model, sentence, label):
    model.zero_grad()

    logit = model(sentence)

    # Turn logit into probability and class
    pred_prob = torch.nn.Sigmoid()(logit).item()
    pred_class = 1 if pred_prob > 0.5 else 0

    reference_indices = token_reference.generate_reference(len(sentence), device=device).unsqueeze(0)

    debugpy.breakpoint()
    attributions_ig, delta = lig.attribute(sentence,
                                            reference_indices,
                                            n_steps=500,
                                            return_convergence_delta=True)

    # print('pred: ', pred_class, f'(probability: {pred_prob:.3f})', ' delta: ', delta)


    add_attributions_to_visualizer(attributions_ig, sentence, pred_prob, pred_class, label, delta, vis_data_records_ig)

def add_attributions_to_visualizer(attributions, sentence, pred_prob, pred_class, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred_prob,
        pred_class,
        label,
        1,
        attributions.sum(),
        vocab.lookup_tokens(sentence[0].cpu().detach().numpy()),
        delta
    ))


def get_attributions(model, sentence, label):
    model.zero_grad()

    logit = model(sentence)

    # Turn logit into probability and class
    pred_prob = torch.nn.Sigmoid()(logit).item()
    pred_class = 1 if pred_prob > 0.5 else 0

    reference_indices = token_reference.generate_reference(len(sentence), device=device).unsqueeze(0)

    attributions_ig, relative_delta = lig.attribute(sentence,
                                            reference_indices,
                                            n_steps=500,
                                            return_convergence_delta=True)

    # print('pred: ', pred_class, f'(probability: {pred_prob:.3f})', ' delta: ', relative_delta)


    # attributions = attributions_ig.sum(dim=2).squeeze(0)
    # attributions = attributions / torch.norm(attributions)
    # attributions = attributions.cpu().detach().numpy()

    attributions = attributions_ig.cpu().detach().numpy()


    return attributions, pred_prob, pred_class, label, relative_delta

    # add_attributions_to_visualizer(attributions_ig, sentence, pred_prob, pred_class, label, delta, vis_data_records_ig)


def get_amino_acid_sentence(sentences):
    letter_sentences = []
    for i, sentence in enumerate(sentences):
        sentence = vocab.lookup_tokens(sentence.detach().numpy())
        letter_sentence = []
        for j, char in enumerate(sentence):
            if not (char == '<pad>' or char == 'END'):
                letter_sentence.append(char)

        letter_sentences.append(letter_sentence)

    return letter_sentences



fname = 'YOUR_CSV.csv'
fpath = fname
with open(fpath, 'r') as f:
    lines = f.readlines()
    args = lines[0].split(',')
    args = [a.strip() for a in args]
    configs = []
    for i in range(1, len(lines)):
        c = lines[i].split(',')
        c = [a.strip() for a in c]
        configs.append(c)

configs = [dict(zip(args, [arg_type_map[arg](config[i])
                            for i, arg in enumerate(args)]))
           for config in configs]

log_dirs = [config['log_dir'] for config in configs]


def get_completed_iters(log_dir):

    completed_folds = []
    for root, dirs, files in os.walk(log_dir):
        for dir in dirs:
            if 'fold' in dir:
                completed_folds.append(dir)

    completed_iters = []
    for fold in completed_folds:
        fold_iters = []
        fold_dir = os.path.join(log_dir, fold)
        for root, dirs, files in os.walk(fold_dir):
            for dir in dirs:
                if 'iter' in dir:
                    fold_iters.append(dir)
        completed_iters.append(fold_iters)

    return completed_folds, completed_iters


vocab, fold_datasets = load_dataset_folds()

for config in configs[10:]:
    model_args = reset_model_args()

    log_dir = config['log_dir']

    completed_folds, completed_iters = get_completed_iters(log_dir)

    config = {**defaults, **config}
    model_args = {arg: config[arg] for arg in model_args}
    model_args['encoder_ansatz'] = ansatz_map[model_args['encoder_ansatz']]
    model_args['recurrent_ansatz'] = ansatz_map[model_args['recurrent_ansatz']]
    if config['embed_type'] == 'random':
        model_args['embed_dim'] = config['embed_dim']


    model = embed_model_map[config['embed_type']](model_args,
                                                            vocab=vocab,
                                                            device=device)



    for fold in completed_folds:
        fold_num = int(fold.split('_')[1])
        train_set, val_set, test_set = fold_datasets[fold_num]

        dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        X_val, y_val = val_set.tensors
        X_test, y_test = test_set.tensors

        X_val, y_val = X_val.to(device), y_val.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        X_val = torch.tensor(X_val).to(device)
        y_val = torch.tensor(y_val).to(device)

        X_test = torch.tensor(X_test).to(device)
        y_test = torch.tensor(y_test).to(device)


        for iter in completed_iters[completed_folds.index(fold)]:
            print(f'fold: {fold} iter: {iter}')

            model_state_dict = torch.load(f'{log_dir}/{fold}/{iter}/best_model.pt', map_location=device)
            # model_state_dict = torch.load(f'{log_dir}/fold_3/iter_2/best_model.pt', map_location=device)

            # Iters > 0 have a different state dixct
            try:
                model.load_state_dict(model_state_dict)
            except RuntimeError:
                for k in ['encoder.q_device.state', 'recurrent_circuit.q_device.state', 'measure.q_device.state']:
                    if k in model_state_dict:
                        model_state_dict.pop(k)

            # # Remove keys it's not expecting
            # for k in ['encoder.q_device.state', 'recurrent_circuit.q_device.state', 'measure.q_device.state']:
            #     if k in model_state_dict:
            #         model_state_dict.pop(k)

            # print(model_state_dict.keys())
            # print(model_state_dict['encoder.q_device.state'])
            # model.load_state_dict(model_state_dict)

            pad_idx = vocab.lookup_indices(['<pad>'])[0]
            token_reference = TokenReferenceBase(reference_token_idx=pad_idx)

            lig = LayerIntegratedGradients(model, model.embedding)

            vis_data_records_ig = []

            # pos_sentences = [X for X, y in zip(X_val, y_val) if y == 1]
            # neg_sentences = [X for X, y in zip(X_val, y_val) if y == 0]
            # pos_y = [y for y in y_val if y == 1]
            # neg_y = [y for y in y_val if y == 0]

            # pos_sentences = [X for X, y in zip(X_val, y_val) if y == 1]
            # neg_sentences = [X for X, y in zip(X_val, y_val) if y == 0]
            # pos_y = [y for y in y_val if y == 1]
            # neg_y = [y for y in y_val if y == 0]
            pos_sentences = [X for X, y in zip(X_test, y_test) if y == 1]
            neg_sentences = [X for X, y in zip(X_test, y_test) if y == 0]
            pos_y = [y for y in y_test if y == 1]
            neg_y = [y for y in y_test if y == 0]

            attributions_pos = []
            attributions_neg = []
            pred_class_pos = []
            pred_class_neg = []
            labels_pos = []
            labels_neg = []

            for s, l in zip(X_test, y_test):
                s = s.unsqueeze(0)
                l = l.item()
                attributions_s, pred_prob, pred_class, label, delta = get_attributions(model, s, l)

                if label == 1:
                    attributions_pos.append(attributions_s)
                    pred_class_pos.append(pred_class)
                    labels_pos.append(label)
                else:
                    attributions_neg.append(attributions_s)
                    pred_class_neg.append(pred_class)
                    labels_neg.append(label)

            attributions_pos = np.array(attributions_pos)
            attributions_neg = np.array(attributions_neg)

            pos_sentences_letters = get_amino_acid_sentence(pos_sentences)
            neg_sentences_letters = get_amino_acid_sentence(neg_sentences)

            import pickle

            results_dict = {'sentences_pos': pos_sentences_letters,
                            'sentences_neg': neg_sentences_letters,
                            'attributions_pos': attributions_pos,
                            'attributions_neg': attributions_neg,
                            'pred_class_pos': pred_class_pos,
                            'pred_class_neg': pred_class_neg,
                            'labels_pos': labels_pos,
                            'labels_neg': labels_neg,
                            'log_dir': log_dir}

            fname = f'ATTRIBUTIONS_DIR/sentence_attributions_{log_dir.split("/")[-1]}_{fold}_{iter}.pkl'
            with open(fname, 'wb') as f:
                pickle.dump(results_dict, f)

            print(f'Saved attributions to {fname}')