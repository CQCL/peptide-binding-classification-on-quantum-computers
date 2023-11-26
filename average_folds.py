"""Read tensorboard logs for k-fold cross-val and average results"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np

from utils.convert_tb import convert_tb_data
from tbparse import SummaryReader

from tensorflow.python.framework.errors_impl import DataLossError


class Processor:
    def __init__(self, file, scalar, rank_by):
        self.file = file
        self.scalar = scalar
        self.rank_by = rank_by

    def get_completed_iters(self):
        """Get structure of file"""

        # Get number of folds, i.e. number of directories containing "fold"
        completed_folds = []
        for item in os.listdir(self.file):
            if os.path.isdir(os.path.join(self.file, item)):
                if "fold" in item:
                    completed_folds.append(item)

        # For each fold get completed iterations
        completed_iters = []
        for fold in completed_folds:
            fold_iters = []
            fold_dir = os.path.join(self.file, fold)
            for item in os.listdir(fold_dir):
                if os.path.isdir(os.path.join(fold_dir, item)):
                    fold_iters.append(item)
            completed_iters.append(fold_iters)

        return completed_folds, completed_iters

    def get_val_f1(self, file, scalar):
        """Get max val f1 from the specified log file"""

        try:
            df = convert_tb_data(file)
            scalar_name = f"{scalar}/f1"
            val_f1 = df[df["name"] == scalar_name]["value"].tolist()

        except DataLossError:
            val_f1 = None

        return val_f1

    def get_f1s(self, folds, iters, scalar):

        all_f1s = []
        for fold, iters in zip(folds, iters):
            fold_f1s = []
            for iter in iters:
                iter_dir = os.path.join(self.file, fold, iter)
                iter_f1s = self.get_val_f1(iter_dir, scalar)
                if iter_f1s is not None:
                    fold_f1s.append(iter_f1s)
            all_f1s.append(fold_f1s)

        return np.array(all_f1s)  # (n_folds, n_iters, n_epochs)

    def get_best_iters(self, max_f1s, completed_iters):
        """
        max_f1s: (n_folds, n_iters)
        """
        max_iters_idx = np.argmax(max_f1s, axis=1)  # (n_folds)
        max_iters = [iters[idx] for iters, idx in zip(completed_iters, max_iters_idx)]
        return max_iters, max_iters_idx

    def get_avg_f1s(self):

        if self.scalar == self.rank_by:
            completed_folds, completed_iters = self.get_completed_iters()
            all_f1s = self.get_f1s(completed_folds, completed_iters, self.scalar)

            # Max over epochs
            max_f1s = np.max(all_f1s, axis=2)  # (n_folds, n_iters)

            max_iters, _ = self.get_best_iters(max_f1s, completed_iters)

            # Max over iters
            best_f1s_over_iters = np.max(max_f1s, axis=1)  # (n_folds)

            # Average over folds
            avg_best_f1 = np.mean(best_f1s_over_iters, axis=0)  # (1)
            std_best_f1 = np.std(best_f1s_over_iters, axis=0)  # (1)

            return avg_best_f1, std_best_f1, completed_folds, max_iters

        else:
            completed_folds, completed_iters = self.get_completed_iters()

            # Get test/val f1 for the iter with best val f1
            all_f1s_scalar = self.get_f1s(completed_folds, completed_iters, self.scalar)
            all_f1s_rank_by = self.get_f1s(
                completed_folds, completed_iters, self.rank_by
            )

            # Max over epochs
            max_f1s_scalar = np.max(all_f1s_scalar, axis=2)  # (n_folds, n_iters)
            max_f1s_rank_by = np.max(all_f1s_rank_by, axis=2)  # (n_folds, n_iters)

            max_iters_rank_by, max_iters_idx_rank_by = self.get_best_iters(
                max_f1s_rank_by, completed_iters
            )

            # Get f1s for each {scalar} fold for {rank_by}'s max iter
            best_f1s_scalar = [
                max_f1s_scalar[fold, ind]
                for fold, ind in enumerate(max_iters_idx_rank_by)
            ]  # (n_folds)

            # Average over folds
            avg_best_f1 = np.mean(best_f1s_scalar, axis=0)  # (1)
            std_best_f1 = np.std(best_f1s_scalar, axis=0)  # (1)

            return avg_best_f1, std_best_f1, completed_folds, max_iters_rank_by

    def get_num_params(self):

        r = SummaryReader(self.file)
        num_params = r.hparams[r.hparams["tag"] == "num_params"]["value"].tolist()[-1]
        return num_params


def get_runs(csv_file):
    """Get list of run names from csv file"""

    df = pd.read_csv(csv_file, sep=",", header=0)
    names = df["log_dir"].tolist()

    return names


def get_existing_log_dirs(log_dir_names):
    """Get list of existing log dirs"""

    existing_dirs = []
    for name in log_dir_names:
        if os.path.exists(name):
            existing_dirs.append(name)

    return existing_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--create-df", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-iters", action="store_true")
    parser.add_argument("--by", type=str, help="Rank iters by val/test")
    parser.add_argument("--suffix", type=str)
    args = parser.parse_args()

    store_df = args.create_df
    plot_ = args.plot
    fname = args.file
    save_iters = args.save_iters
    rank_by = args.by
    suffix = args.suffix

    assert not (args.val and args.test)
    if args.val:
        if rank_by == "val":
            scalar = "val"
        elif rank_by == "test":
            raise ValueError("Cannot rank by test if using val data")
    elif args.test:
        scalar = "test"

    names = get_runs(fname)
    existing_names = get_existing_log_dirs(names)

    all_runs = []
    best_avg_f1s = []
    std_dev_runs = []
    num_params_list = []
    best_iters_list = []
    folds_list = []
    for name in existing_names:
        p = Processor(name, scalar, rank_by)

        avg_best_f1, std_dev, completed_folds, max_iters = p.get_avg_f1s()
        num_params = p.get_num_params()

        best_avg_f1s.append(avg_best_f1)
        std_dev_runs.append(std_dev)
        num_params_list.append(num_params)
        best_iters_list.append(max_iters)
        folds_list.append(completed_folds)

    print(list(zip(existing_names, best_avg_f1s, std_dev_runs)))

    if save_iters:
        best_iters_dict = {
            "log_dir": existing_names,
            "folds": folds_list,
            "best_iters": best_iters_list,
        }
        with open(f"RERUN_tq_{scalar}.pkl", "wb") as f:
            pickle.dump(best_iters_dict, f)

    if store_df:
        df = pd.DataFrame(
            {
                "log_dir": existing_names,
                "best_f1": best_avg_f1s,
                "std_dev": std_dev_runs,
                "num_params": num_params_list,
            }
        )
        df.to_csv(
            f"pub_test_{suffix}_{scalar}_rankby_{rank_by}_df.csv",
            index=False,
        )
