import random
import sys
from tqdm import tqdm
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from model.ansatz import *
from model.select_model import *
from model.utils import *

arg_type_map = {'embed_type': str, 'embed_dim': int, 'encoder_ansatz': str,
                'encoder_layers': int, 'recurrent_ansatz': str,
                'recurrent_layers': int, 'reupload_count': int,
                'n_wires': int, 'measure': str, 'learning_rate': float,
                'batch_size': int, 'epochs': int, 'weight_decay': float,
                'dropout_prob': float, 'seed': int, 'use_gpu': str2bool,
                'log_dir': str, 'measure': str, 'final': str, 'dataset': str}

model_args = {'encoder_ansatz', 'encoder_layers', 'recurrent_ansatz',
              'recurrent_layers', 'reupload_count', 'n_wires', 'measure',
              'dropout_prob', 'final'}
required = {'embed_type', 'encoder_ansatz', 'n_wires', 'log_dir'}
defaults = {'embed_dim': 0, 'encoder_layers': 1, 'recurrent_ansatz': 'None',
            'recurrent_layers': 1, 'reupload_count': 1, 'measure': 'all',
            'learning_rate': 1e-2, 'batch_size': 16, 'epochs': 100,
            'weight_decay': 0.0, 'dropout_prob': 0.0, 'final': 'linear',
            'use_gpu': torch.cuda.is_available(), 'dataset': 'default'}

embed_model_map = {'pure': get_pure_qrnn_kfold, 'single': get_single_embed_qrnn,
                   'random': get_random_embed_qrnn_kfold,
                   'position': get_position_embed_qrnn}
ansatz_map = {'None': None, 'sim7': sim7, 'sim8': sim8, 'sim9': sim9,
              'sim11': sim11, 'sim12': sim12, 'sim13': sim13, 'sim14': sim14,
              'sim17': sim17}
dataset_map = {'default': load_dataset_folds, 'no_end': load_dataset_folds_no_end_token,
               'bigrams': load_dataset_folds_bigrams}

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


def train(dataloader):
    model.train()
    epoch_loss = 0.

    for step, batch in enumerate(dataloader):
        d, l = batch
        d, l = d.to(device), l.to(device)
        logits = model(d).squeeze()
        loss = loss_fn(logits, l.float())
        epoch_loss += loss.item()

        writer.add_scalar('train/step_loss', loss.item(),
                          epoch * len(dataloader) + step + 1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss /= len(dataloader)
    return epoch_loss


def eval(X, y):
    model.eval()

    with torch.no_grad():
        logits = model(X).squeeze()
        # val_loss = loss_fn(logits, y.float())
        val_acc = acc(logits, y)
        val_f1 = f1(logits, y)
        # val_confusion = confusion(logits, y)

    # return val_loss, val_acc, val_f1, val_confusion
    return val_acc, val_f1


for i, config in enumerate(configs):
    config = {**defaults, **config}
    print(f'Running config: {i}')

    vocab, fold_datasets = dataset_map[config['dataset']]()

    outer_log_dir = f'{config["log_dir"]}_{config["dataset"]}'

    if config['use_gpu']:
        device = get_idle_gpu()
    else:
        device = torch.device('cpu')

    if not all(arg in config for arg in required):
        warnings.warn(f'Config {i} is missing required arguments.')
        continue

    print("Loading data and initialising model...")

    model_args = {arg: config[arg] for arg in model_args}
    model_args['encoder_ansatz'] = ansatz_map[model_args['encoder_ansatz']]
    model_args['recurrent_ansatz'] = ansatz_map[model_args['recurrent_ansatz']]
    if config['embed_type'] == 'random':
        model_args['embed_dim'] = config['embed_dim']

    # fold_results = []
    for fold, (train_set, val_set, test_set) in enumerate(fold_datasets):
        print(f'Running fold: {fold}')

        fold_log_dir = os.path.join(outer_log_dir, f'fold_{fold}')

        print('Loading data...')

        X_train, y_train = train_set.tensors
        X_val, y_val = val_set.tensors
        X_test, y_test = test_set.tensors

        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        for seed in [2, 4]:
            iter_dir = os.path.join(fold_log_dir, f'iter_{seed}')

            print(f'Running seed: {seed}')

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            print('Initialising model and dataloaders...')

            model = embed_model_map[config['embed_type']](model_args,
                                                          vocab,
                                                          device=device)
            config['num_params'] = sum(p.numel() for p in model.parameters())

            print("Making log directory...")

            try:
                os.makedirs(iter_dir)
            except FileExistsError:
                warnings.warn(f'Skipping config {i} as log directory '
                              f'{iter_dir} already exists.')
                continue

            dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)

            # writer = SummaryWriter(log_dir=config['log_dir'])
            writer = SummaryWriter(log_dir=iter_dir)

            print("Initialising optimiser...")

            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=config['learning_rate'],
                                        weight_decay=config['weight_decay'])
            if config['final'] == 'linear':
                loss_fn = torch.nn.BCEWithLogitsLoss()
                acc = torch_binary_logits_acc
                f1 = torch_logits_f1
                confusion = torch_logits_confusion
            elif config['final'] == 'rescale':
                loss_fn = torch.nn.BCELoss()
                acc = torch_binary_acc
                f1 = torch_f1
                confusion = torch_confusion

            print("Training model...")

            max_val_acc = 0.
            max_val_f1 = 0.

            for epoch in tqdm(range(config['epochs'])):
                train_loss = train(dataloader)
                train_acc, train_f1 = eval(X_train, y_train)
                # val_loss, val_acc, val_f1, val_confusion = eval(X_val, y_val)
                val_acc, val_f1 = eval(X_val, y_val)

                if val_acc > max_val_acc:
                    max_val_acc = val_acc

                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
                    # Make checkpoint
                    torch.save(model.state_dict(), os.path.join(iter_dir, 'best_model.pt'))
                    # Save confusion matrix
                    # np.save(os.path.join(fold_log_dir, 'confusion.npy'), val_confusion)

                writer.add_scalar('train/epoch_loss', train_loss, epoch + 1)
                writer.add_scalar('train/acc', train_acc, epoch + 1)
                writer.add_scalar('train/f1', train_f1, epoch + 1)
                writer.add_scalar('val/acc', val_acc, epoch + 1)
                writer.add_scalar('val/max_acc', max_val_acc, epoch + 1)
                writer.add_scalar('val/f1', val_f1, epoch + 1)
                writer.add_scalar('val/max_f1', max_val_f1, epoch + 1)

            model.load_state_dict(torch.load(os.path.join(iter_dir, 'best_model.pt')))
            # test_loss, test_acc, test_f1, test_confusion = eval(X_test, y_test)
            test_acc, test_f1 = eval(X_test, y_test)

            writer.add_scalar('test/acc', test_acc, epoch + 1)
            writer.add_scalar('test/f1', test_f1, epoch + 1)

            writer.add_hparams(config, {'test_acc': test_acc,
                                        'test_f1': test_f1})

            print("Finished training.")
            print(f"Max val f1 for fold {fold}: {max_val_f1}")
            # fold_results.append(max_val_f1)


        # avg_fold_results = np.mean(fold_results)
        # print(f"Average fold results: {avg_fold_results}")



        # print("Finished training.")

        # print("Evaluating model...")

        # #save model
        # torch.save(model.state_dict(), config['log_dir'] + '/model.pt')
        # #save vocab
        # np.save(config['log_dir'] + '/vocab.npy', vocab.get_itos())

        # # Save test data
        # np.save(config['log_dir'] + '/X_test.npy', X_val.cpu().numpy())
        # np.save(config['log_dir'] + '/y_test.npy', y_val.cpu().numpy())

        # params_dict = {}
        # params_dict.update(config)
        # weights = model.state_dict()

        # params_dict.update('params': model.parameters())






        # Print embeddings



        # for p in model.parameters():
        #     if p.requires_grad:
        #         # If param has name
        #         if p.name is not None:
        #             print(p.name)
        #         # If param has no name
        #         print(p)
        #     # print(p)

        # print("------------------")


        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        # test()

