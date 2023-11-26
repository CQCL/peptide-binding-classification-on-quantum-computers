# Ansatzes from this paper: https://arxiv.org/pdf/1905.10876.pdf
import itertools

def sim7(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'rx', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crz', 'wires': [i, i+1]} for i in range(0, n_wires//2 * 2, 2)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rx', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crz', 'wires': [i, i+1]} for i in range(1, (n_wires-1)//2 * 2 + 1, 2)])
    return enc, next(counter)


def sim8(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'rx', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, i+1]} for i in range(0, n_wires//2 * 2, 2)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rx', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, i+1]} for i in range(1, (n_wires-1)//2 * 2 + 1, 2)])
    return enc, next(counter)


def sim9(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'hadamard', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': None, 'func': 'cz', 'wires': [i, i + 1]} for i in range(n_wires - 1)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rx', 'wires': [i]} for i in range(n_wires)])
    return enc, next(counter)


def sim11(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i % n_wires]} for i in range(n_wires)])
        enc.extend([{'input_idx': None, 'func': 'cnot', 'wires': [i, i+1]} for i in range(0, n_wires//2 * 2, 2)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(1, (n_wires-1)//2 * 2 + 1)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i]} for i in range(1, (n_wires-1)//2 * 2 + 1)])
        enc.extend([{'input_idx': None, 'func': 'cnot', 'wires': [i, i+1]} for i in range(1, (n_wires-1)//2 * 2 + 1, 2)])
    return enc, next(counter)


def sim12(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i % n_wires]} for i in range(n_wires)])
        enc.extend([{'input_idx': None, 'func': 'cz', 'wires': [i, i+1]} for i in range(0, n_wires//2 * 2, 2)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(1, (n_wires-1)//2 * 2 + 1)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i]} for i in range(1, (n_wires-1)//2 * 2 + 1)])
        enc.extend([{'input_idx': None, 'func': 'cz', 'wires': [i, i+1]} for i in range(1, (n_wires-1)//2 * 2 + 1, 2)])
    return enc, next(counter)


def sim13(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crz', 'wires': [i, (i - 1) % n_wires]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crz', 'wires': [i, (i + 1) % n_wires]} for i in [0] + list(range(n_wires - 1, 0, -1))])
    return enc, next(counter)


def sim14(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, (i - 1) % n_wires]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, (i + 1) % n_wires]} for i in [0] + list(range(n_wires - 1, 0, -1))])
    return enc, next(counter)


def sim17(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'rx', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'rz', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crz', 'wires': [i, i+1]} for i in range(0, n_wires//2 * 2, 2)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crz', 'wires': [i, i+1]} for i in range(1, (n_wires-1)//2 * 2 + 1, 2)])
    return enc, next(counter)