import copy
import random
import sys
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter

from model.ansatz import *
from model.select_classical_model import *

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

embed_model_map = {'random': get_random_classical_rnn,
                   'pretrained': get_pretrained_classical_rnn}


def train(epoch, model, dataloader, writer, loss_fn, optimizer):
    model.train()

    epoch_loss = 0
    for step, batch in enumerate(dataloader):
        d, d_lens, l = batch
        logits = model(d, d_lens).squeeze()
        loss = loss_fn(logits, l.float())
        epoch_loss += loss.item()
        writer.add_scalar('train/step_loss', loss.item(),
                          epoch * len(dataloader) + step + 1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss


def eval(model, dataloader):
    model.eval()

    for X, X_lens, y in dataloader:
        acc = torch_binary_logits_acc(model(X, X_lens), y)
        f1 = torch_logits_f1(model(X, X_lens), y)
        break

    return acc, f1


filepath = sys.argv[1]

with open(filepath, 'r') as f:
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

vocab, fold_datasets = load_dataset_folds()

for i, shared_config in enumerate(configs):
    shared_config = {**defaults, **shared_config}
    print(f'Running config: {i}')

    log_dir = shared_config['log_dir']

    if shared_config['use_gpu']:
        device = get_idle_gpu()
    else:
        device = torch.device('cpu')

    for fold, datasets in enumerate(fold_datasets):
        print(f'Running fold: {fold}')

        fold_dir = os.path.join(log_dir, f'fold_{fold}')

        for seed in [0, 2, 4]:
            print(f'Running with seed: {seed}')

            config = copy.copy(shared_config)
            iter_dir = os.path.join(fold_dir, f'iter_{seed}')

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            print("Loading data and initialising model...")

            model_args = {arg: config[arg] for arg in model_args}
            if config['embed_type'] == 'random':
                model_args['embed_dim'] = config['embed_dim']

            model, train_loader, train_eval_loader, val_loader, test_loader = \
                get_random_classical_rnn_kfold(model_args,
                                               datasets,
                                               vocab,
                                               batch_size=config['batch_size'],
                                               device=device)
            config['num_params'] = sum(p.numel() for p in model.parameters())

            print("Making log directory...")

            try:
                os.makedirs(iter_dir)
            except FileExistsError:
                warnings.warn(f'Skipping config {i} as log directory '
                              f'{iter_dir} already exists.')
                continue

            writer = SummaryWriter(log_dir=iter_dir)

            print("Initialising optimiser...")

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config['learning_rate'],
                                         weight_decay=config['weight_decay'])
            loss_fn = torch.nn.BCEWithLogitsLoss()

            print("Training model...")

            max_val_f1 = 0.
            max_val_acc = 0.
            for epoch in range(config['epochs']):
                train_loss = train(epoch, model, train_loader, writer,
                                   loss_fn, optimizer)
                train_acc, train_f1 = eval(model, train_eval_loader)
                val_acc, val_f1 = eval(model, val_loader)

                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
                    torch.save(model.state_dict(), os.path.join(iter_dir, 'best_model.pt'))

                if val_acc > max_val_acc:
                    max_val_acc = val_acc

                writer.add_scalar('train/epoch_loss', train_loss, epoch + 1)
                writer.add_scalar('train/acc', train_acc, epoch + 1)
                writer.add_scalar('train/f1', train_f1, epoch + 1)
                writer.add_scalar('val/acc', val_acc, epoch + 1)
                writer.add_scalar('val/max_acc', max_val_acc, epoch + 1)
                writer.add_scalar('val/f1', val_f1, epoch + 1)
                writer.add_scalar('val/max_f1', max_val_f1, epoch + 1)

            model.load_state_dict(torch.load(os.path.join(iter_dir, 'best_model.pt')))
            test_acc, test_f1 = eval(model, test_loader)

            writer.add_hparams(config, {'test/acc': test_acc,
                                        'test/f1': test_f1})

            print("Finished training.")