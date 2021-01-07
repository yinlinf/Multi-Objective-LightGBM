import argparse
from typing import List, Dict, Optional

import hypertune
import lightgbm as lgb
import logging
import numpy as np
import sys

from d_lgbm.train_valid_file_combiner import LightGBMTrainValidFileCombiner
import re
import subprocess
from pathlib import Path
import pandas

_DATA_DIR = "/tmp/data"
_DATASET_DIR = "tmp/dataset"

_LOCAL_BZ_FEATURES_DIR = f"{_DATA_DIR}/features"
_PARTITION_FILE_PTN = r"part-(\d+)"
_LOCAL_TREE_CONFIG_PATH = f"{_DATA_DIR}/config/tree_config.conf"

_LOCAL_TRAIN_FILE = f"{_DATASET_DIR}/features.train"
_LOCAL_VALID_FILE = f"{_DATASET_DIR}/features.test"

_METRIC_TAG = "validation_ndcg"
_NOTHING_LABEL = 0.0
_PURCHASE_LABEL = 4.0
MAX_NDCG_POS = 48
N_SIGMOID_BINS = 1024 * 1024
MIN_SIGMOID_ARG = -25
MAX_SIGMOID_ARG = 25

from d_lgbm.utils import sparse_vector_string_extract_column
import collections
# sys.path.insert(0, './')
# from  moltr.calculator import Calculator, MIN_SIGMOID_ARG, MAX_SIGMOID_ARG
from lambdaobj import get_gradients

def get_query_boundaries(groups):
    assert(len(groups) > 0)
    query_boundaries = [0] + list(np.cumsum(groups))
    return query_boundaries

class Calculator:
    def __init__(self, gains, groups, k):
        self.query_boundaries = get_query_boundaries(groups)
        self.gains = gains
        self.k = k
        self.discounts = Calculator._fill_discount_table(np.max(groups), k)

        # print("Computing inverse_max_dcg-s..")
        self.inverse_max_dcgs = Calculator._fill_inverse_max_dcg_table(
            self.gains,
            self.query_boundaries,
            self.discounts,
            k
        )
        # print("Computing sigmoids..")
        self.sigmoids, self.sigmoid_idx_factor = Calculator._fill_sigmoid_table(
            N_SIGMOID_BINS,
            MIN_SIGMOID_ARG,
            MAX_SIGMOID_ARG
        )

    def get_sigmoid(self, score):
        if score <= MIN_SIGMOID_ARG:
            return self.sigmoids[0]
        elif score >= MAX_SIGMOID_ARG:
            return self.sigmoids[-1]
        else:
            return self.sigmoids[int((score - MIN_SIGMOID_ARG) * self.sigmoid_idx_factor)]

    def compute_ndcg(self, scores):
        dcgs = np.zeros(len(self.query_boundaries) - 1)

        for i in range(len(self.query_boundaries) - 1):
            order = np.argsort(scores[self.query_boundaries[i]:self.query_boundaries[i + 1]])[::-1]
            g = np.array(self.gains[self.query_boundaries[i]:self.query_boundaries[i + 1]])[order][:self.k]
            dcgs[i] = np.sum(g * self.discounts[1:(len(g) + 1)])
        return np.mean(dcgs * self.inverse_max_dcgs)

    @staticmethod
    def _fill_discount_table(max_group_length, k):
        discounts = np.zeros(1 + max_group_length)
        m = min(max_group_length, k)
        discounts[1:(1 + m)] = 1 / np.log2(1 + np.arange(1, m + 1))
        return discounts

    @staticmethod
    def _fill_inverse_max_dcg_table(gains, query_boundaries, discounts, k):
        inverse_max_dcgs = np.zeros(len(query_boundaries) - 1)

        for i in range(len(query_boundaries) - 1):
            g = np.sort(gains[query_boundaries[i]:query_boundaries[i + 1]])[::-1][:k]
            assert(len(discounts) > len(g))
            max_dcg = np.sum(g * discounts[1:(len(g) + 1)])
            assert(max_dcg > 0)
            inverse_max_dcgs[i] = 1 / max_dcg
        return inverse_max_dcgs

    @staticmethod
    def _fill_sigmoid_table(n_sigmoid_bins, min_sigmoid_arg, max_sigmoid_arg):
        sigmoid_idx_factor = n_sigmoid_bins / (max_sigmoid_arg - min_sigmoid_arg)
        sigmoids = 2.0 / (1 + np.exp(2.0 *
                                     (np.arange(n_sigmoid_bins)
                                      / sigmoid_idx_factor + min_sigmoid_arg)))

        return sigmoids, sigmoid_idx_factor

def set_group_for_dataset(data_path, query_id_column):
    with open(data_path, 'r') as fid:
        lines = fid.readlines()
    query_ids = []
    for line_idx, line in enumerate(lines):
        line_query_id = int(sparse_vector_string_extract_column(line, query_id_column))
        query_ids += [line_query_id]
    count = collections.Counter(query_ids)
    groups = list(count.values())
    data = lgb.Dataset(data_path, free_raw_data=False)
    data.set_group(groups)
    return data


def create_new_labels(dataset:lgb.Dataset):
    dataset.construct()
    labels = np.array(dataset.get_label())

    labels_purchase = np.zeros(len(labels))
    labels_purchase[labels == 1.0] = 1.0
    labels_purchase[labels == 2.0] = 3.0
    labels_purchase[labels == 3.0] = 7.0
    labels_purchase[labels == 4.0] = 15.0
    dataset.labels_purchase = labels_purchase
    logging.info(f"Frequency of purchase labels:{collections.Counter(labels_purchase)}")

    labels_click = np.zeros(len(labels))
    labels_click[labels != _NOTHING_LABEL] = 1.0
    dataset.labels_click = labels_click
    logging.info(f"Frequency of click labels:{collections.Counter(labels_click)}")
    return dataset

def get_grad_hess(labels, preds, groups, calculator):
    grad = np.zeros(len(preds))
    hess = np.zeros(len(preds))
    get_gradients(np.ascontiguousarray(labels, dtype=np.double),
                  np.ascontiguousarray(preds),
                  len(preds),
                  np.ascontiguousarray(groups),
                  np.ascontiguousarray(calculator.query_boundaries),
                  len(calculator.query_boundaries) - 1,
                  np.ascontiguousarray(calculator.discounts),
                  np.ascontiguousarray(calculator.inverse_max_dcgs),
                  np.ascontiguousarray(calculator.sigmoids),
                  len(calculator.sigmoids),
                  MIN_SIGMOID_ARG,
                  MAX_SIGMOID_ARG,
                  calculator.sigmoid_idx_factor,
                  np.ascontiguousarray(grad),
                  np.ascontiguousarray(hess))
    return grad, hess


def combined_objective(preds, dataset):
    groups = dataset.get_group()
    if len(groups) == 0:
        raise Error("Group/query data should not be empty.")
    else:
        grad_1, hess_1 = get_grad_hess(
            dataset.labels_purchase, preds, groups, dataset.calculator_1
        )
        grad_2, hess_2 = get_grad_hess(
            dataset.labels_click, preds, groups, dataset.calculator_2
        )
        alpha = dataset.alpha
        return alpha * grad_1 + (1 - alpha) * grad_2, alpha * hess_1 + (1 - alpha) * hess_2

def combined_eval(preds, dataset):
    ndcg_1 = dataset.calculator_1.compute_ndcg(preds)
    ndcg_2 = dataset.calculator_2.compute_ndcg(preds)
    is_higher_better = True
    return [("ndcg_1", ndcg_1, is_higher_better),
            ("ndcg_2", ndcg_2, is_higher_better)]

def DatasetWithTwoLabels(dataset, alpha):
    dataset.calculator_1 = Calculator(dataset.labels_purchase, dataset.get_group(), 10)
    dataset.calculator_2 = Calculator(dataset.labels_click, dataset.get_group(), 10)
    dataset.alpha = alpha
    return dataset


def _run_shell_command(command:str):
    command_parts = command.split()
    subprocess.run(command_parts)

def _create_local_directories():
    for data_subpath in ("features", "config"):
        (Path(_DATA_DIR) / data_subpath).mkdir(parents=True, exist_ok=True)

def _copy_tree_config_locally(gcs_tree_config_path, local_tree_config_path):
    _run_shell_command(f"gsutil cp {gcs_tree_config_path} {local_tree_config_path}")

def _copy_bz_features_locally(gcs_bz_features_path, local_bz_features_path):
    _run_shell_command(f"gsutil -m cp {gcs_bz_features_path} {local_bz_features_path}/")


def _parse_tree_config_file(config_path):
    with open(config_path, 'r') as fid:
        lines = [l.strip() for l in fid]
        lines = [l for l in lines if not l.startswith('#') and '=' in l]
    params = {}
    for line in lines:
        lhs, rhs = re.split(r"\s*=\s*", line)
        params[lhs] = rhs
    return params

def _create_train_valid_files(train_ratio,
                              query_id_column):

    Path(_DATASET_DIR).mkdir(parents=True, exist_ok=True)
    train_valid_creator = LightGBMTrainValidFileCombiner(train_ratio,
                                                         query_id_column,
                                                         _PARTITION_FILE_PTN,
                                                         _LOCAL_TRAIN_FILE,
                                                         _LOCAL_VALID_FILE)
    train_valid_creator.create_train_valid_files(_LOCAL_BZ_FEATURES_DIR)

def _get_best_eval_result(evals_result):
    scores = evals_result['valid_0']['ndcg@10']
    best_idx = np.argmax(scores)
    epoch = best_idx + 1
    return [epoch,
           evals_result['valid_0']['ndcg@10'][best_idx],
           # evals_result['valid_0']['ndcg_1'][best_idx],
           evals_result['valid_0']['ndcg_2'][best_idx]]

def _report_metric(hpt: hypertune.HyperTune,
                   epoch,
                   metric_value):

    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag=_METRIC_TAG,
        metric_value=metric_value,
        global_step=epoch)

def _make_validation_labels_purchase_only(valid_ds:lgb.Dataset):
    valid_ds.construct()
    labels = np.array(valid_ds.get_label())
    non_purchase = (labels != _PURCHASE_LABEL)
    non_purchase_interaction = np.logical_and(non_purchase, labels != _NOTHING_LABEL)
    logging.info(f"Number of non-purchase interactions in valid: {non_purchase_interaction.sum()}")
    logging.info(f"Number of total non-purchases in valid: {non_purchase.sum()}")
    labels[non_purchase] = 0.0
    valid_ds.set_label(labels)

def _train_model_report_metrics(tree_params,
                                make_validation_labels_purchase_only):
    train_ds = set_group_for_dataset(_LOCAL_TRAIN_FILE, query_id_column)
    valid_ds = set_group_for_dataset(_LOCAL_VALID_FILE, query_id_column)
    create_new_labels(train_ds)
    create_new_labels(valid_ds)

    if make_validation_labels_purchase_only:
        _make_validation_labels_purchase_only(valid_ds)

    alpha_values = np.arange(0.0, 1.1, 0.1)
    best_eval_result = []
    for alpha in alpha_values:
        evals_result = {}
        train_data = DatasetWithTwoLabels(train_ds, alpha)
        valid_data = DatasetWithTwoLabels(valid_ds, alpha)
        model =lgb.train(params=tree_params,
                         train_set=train_data,
                         valid_sets=[valid_data],
                         fobj=combined_objective,
                         feval=combined_eval,
                         # callbacks=[lgb.print_evaluation()],
                         evals_result=evals_result)
        best_eval_result.append(_get_best_eval_result(evals_result))
    df = pandas.DataFrame(zip(alpha_values, best_eval_result))
    print(df)

    eval_scores = evals_result['valid_0']['ndcg_1']

    hpt = hypertune.HyperTune()
    for idx,score in enumerate(eval_scores):
        epoch = idx + 1
        _report_metric(hpt, epoch, score)


def _parse_extra_arg_params(args_list:List[str]) -> Dict[str, str]:
    """
    :param args_list: Argument list passed from command line. Assumed to alternate between
        argument names prefixed by "--", and argument values e.g
        --num_trees 300
        Or to be in format --num_trees=300
    """
    params = {}
    for i,arg in enumerate(args_list):
        arg_equal_match = re.match("--([^=]+)=(.+)", arg)
        if arg_equal_match:
            arg_name, arg_value = arg_equal_match.groups()
        elif arg.startswith("--"):
            arg_name = arg[2:]
            arg_value = args_list[i+1]
        else:
            continue
        params[arg_name] = arg_value

    return params

def _parse_tree_params(tree_config_path, extra_args) -> Dict[str,str]:
    tree_params = _parse_tree_config_file(tree_config_path)
    logging.info(f"Parsed tree params: {tree_params}")

    logging.info(f"Extra args: {extra_args}")
    extra_params = _parse_extra_arg_params(extra_args)
    logging.info(f"Extra params: {extra_params}")

    tree_params.update(extra_params)
    logging.info(f"Final params: {tree_params}")
    return tree_params

def _extract_query_id_column(tree_params: Dict[str,str]) -> Optional[int]:
    for column_alias in ("group", "group_id","query_column","query","query_id"):
        if column_alias in tree_params:
            return int(tree_params[column_alias])
    return None

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--bz-features-path")
    ap.add_argument("--tree-config-path")
    ap.add_argument("--train-ratio", default=0.9)
    ap.add_argument("--make-validation-ndcg-purchase-only", action="store_true")
    args, extra_args = ap.parse_known_args()

    logging.basicConfig(stream=sys.stderr)

    _create_local_directories()
    _copy_tree_config_locally(args.tree_config_path, _LOCAL_TREE_CONFIG_PATH)
    tree_params = _parse_tree_params(_LOCAL_TREE_CONFIG_PATH, extra_args)

    query_id_column = _extract_query_id_column(tree_params)

    _copy_bz_features_locally(args.bz_features_path, _LOCAL_BZ_FEATURES_DIR)
    _create_train_valid_files(args.train_ratio, query_id_column)
    _train_model_report_metrics(tree_params,
                                args.make_validation_ndcg_purchase_only)


