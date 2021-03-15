import argparse
from typing import List, Dict, Optional
import hypertune
import lightgbm as lgb
import logging
import sys
from d_lgbm.train_valid_file_combiner import LightGBMTrainValidFileCombiner
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas

# from multi_objectives import set_group_for_dataset
# from multi_objectives import customized_objective_click, customized_eval_click, report_metrics_1
# from multi_objectives import customized_objective_price, customized_eval_price, report_metrics_2

_DATA_DIR = "tmp/data"
_DATASET_DIR = "tmp/dataset"

_LOCAL_BZ_FEATURES_DIR = f"{_DATA_DIR}/features"
_LOCAL_BZ_FEATURES_DIR_TEST = f"{_DATA_DIR}/features_test"
_PARTITION_FILE_PTN = r"part-(\d+)"
_LOCAL_TREE_CONFIG_PATH = f"{_DATA_DIR}/config/tree_config.conf"
_LOCAL_MODEL_PATH = f"{_DATA_DIR}/model"

_LOCAL_TRAIN_FILE = f"{_DATASET_DIR}/features.train"
_LOCAL_VALID_FILE = f"{_DATASET_DIR}/features.valid"
_LOCAL_TEST_FILE = f"{_DATASET_DIR}/features.test"
_LOCAL_TEST_FILE_NULL = f"{_DATASET_DIR}/features.test_null"
daysSinceOriginalCreate_id_column = 28
import numpy as np
import lightgbm as lgb

from sklearn import metrics
from d_lgbm.utils import sparse_vector_string_extract_column
import collections
from lambdaobj import get_gradients

_METRIC_TAG = "validation_ndcg"
_NOTHING_LABEL = 0.0
_CLICK_LABEL = 1.0
_FAV_LABEL = 2.0
_CART_LABEL = 3.0
_PURCHASE_LABEL = 4.0
MAX_NDCG_POS = 10
N_SIGMOID_BINS = 1024 * 1024
MIN_SIGMOID_ARG = -25
MAX_SIGMOID_ARG = 25
price_id_column = 69

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

def set_group_for_dataset(data_path, query_id_column):
    # get group information and add to data object for later use
    last_query_id = None
    group_size = 0
    groups = []
    days_created = []
    missing_days = []
    for line in open(data_path, 'r'):
        line_query_id = int(sparse_vector_string_extract_column(line, query_id_column))
        group_size += 1
        if last_query_id != line_query_id:
            if last_query_id is not None:
                groups.append(group_size - 1)
            last_query_id = line_query_id
            group_size = 1

        line_days = sparse_vector_string_extract_column(line, daysSinceOriginalCreate_id_column)
        if line_days is None:
            missing_days.append(line_query_id)
            line_days = 180
        days_created.append(line_days)

    groups.append(group_size)
    np.asarray(groups, dtype=np.uint8)
    days_created = [7-x if x < 7 else 0.5 for x in days_created]
    np.asarray(days_created, dtype=np.float16)
    data = lgb.Dataset(data_path, free_raw_data=False)
    data.set_group(groups)
    data.labels_daysCreated = days_created
    return data

def set_price_label_for_dataset(data, price_id_column):
    prices = []
    missing_prices = []
    for line in open(data_path, 'r'):
        line_price = sparse_vector_string_extract_column(line, price_id_column)
        if line_price is None:
            missing_prices.append(line_query_id)
            line_price = 0.0001
        log_price = np.log2(1 + line_price)
        prices.append(log_price)
    # print('Listings Missing Price:', collections.Counter(missing_prices))
    print('Percent of Missing Price: %.2f %%' % (len(missing_prices) / len(prices) * 100))
    np.asarray(prices, dtype=np.float16)
    data.labels_price = prices
    return data

def set_days_label_for_dataset(data, daysSinceOriginalCreate_id_column):
    days_created = []
    missing_days = []
    for line in open(data_path, 'r'):
        line_query_id = int(sparse_vector_string_extract_column(line, query_id_column))
        line_days = sparse_vector_string_extract_column(line, daysSinceOriginalCreate_id_column)
        if line_days is None:
            missing_days.append(line_query_id)
            line_days = 0
        days_created.append(line_days)
    print('Listings Missing Days:', collections.Counter(missing_days))
    print('Percent of Missing Days: %.2f %%' % (len(missing_days) / len(days_created) * 100))
    np.asarray(days_created, dtype=np.float16)
    # print(collections.Counter(days_created))
    data.labels_daysCreated = days_created
    return data


def customized_objective_freshness(preds, dataset):
    # define customized objective function as
    # alpha * purchase_ndcg + (1 - alpha) * click_ndcg
    groups = dataset.get_group()
    calculator_1 = Calculator(dataset.get_label(), groups, MAX_NDCG_POS)
    calculator_2 = Calculator(dataset.labels_daysCreated, groups, MAX_NDCG_POS)
    if len(groups) == 0:
        raise Error("Group/query data should not be empty.")
    else:
        grad_1, hess_1 = get_grad_hess(
            dataset.get_label(), preds, groups, calculator_1
        )
        grad_2, hess_2 = get_grad_hess(
            dataset.labels_daysCreated, preds, groups, calculator_2
        )
        alpha = dataset.alpha
        return alpha * grad_1 + (1 - alpha) * grad_2, alpha * hess_1 + (1 - alpha) * hess_2

def customized_eval_freshness(preds, dataset):
    # define customized evaluation function
    dataset.construct()
    groups = dataset.get_group()
    calculator_1 = Calculator(dataset.get_label(), groups, 10)
    calculator_2 = Calculator(dataset.labels_daysCreated, groups, 10)
    ndcg_1 = calculator_1.compute_ndcg(preds)
    ndcg_2 = calculator_2.compute_ndcg(preds)
    alpha = dataset.alpha
    combined_ndcg = alpha * ndcg_1 + (1 - alpha) * ndcg_2
    return [("combined_ndcg", combined_ndcg, True)]

def report_metrics_freshness(preds, dataset):
    # define customized metrics for test data
    dataset.construct()
    groups = dataset.get_group()
    labels = dataset.get_label()
    labels_purchase = np.zeros(len(labels), dtype=np.uint8)
    labels_purchase[labels == 4.0] = 1.0 # purchase only
    calculator_1 = Calculator(labels_purchase, groups, 10)
    calculator_2 = Calculator(dataset.labels_daysCreated, groups, 10)
    ndcg_1 = calculator_1.compute_ndcg(preds)
    ndcg_2 = calculator_2.compute_ndcg(preds)
    return [ndcg_1, ndcg_2]


def _run_shell_command(command:str):
    command_parts = command.split()
    subprocess.run(command_parts)

def _create_local_directories():
    for data_subpath in ("features", "config", "features_test", "model"):
        (Path(_DATA_DIR) / data_subpath).mkdir(parents=True, exist_ok=True)

def _copy_tree_config_locally(gcs_tree_config_path, local_tree_config_path):
    _run_shell_command(f"gsutil cp {gcs_tree_config_path} {local_tree_config_path}")

def _copy_bz_features_locally(gcs_bz_features_path, local_bz_features_path):
    _run_shell_command(f"gsutil -m cp {gcs_bz_features_path} {local_bz_features_path}/")

def _copy_model_locally(pretrained_model_path, local_model_path):
    _run_shell_command(f"gsutil -m cp -r {pretrained_model_path} {local_model_path}/")

def _copy_model_to_gcs(post_trained_model_path, gcs_model_path):
    _run_shell_command(f"gsutil -m cp {post_trained_model_path} {gcs_model_path}/")

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
    # prepare for testing dataset
    train_valid_creator = LightGBMTrainValidFileCombiner(1.0,
                                                        query_id_column,
                                                        _PARTITION_FILE_PTN,
                                                        _LOCAL_TEST_FILE,
                                                        _LOCAL_TEST_FILE_NULL)
    train_valid_creator.create_train_valid_files(_LOCAL_BZ_FEATURES_DIR_TEST)

def _get_best_eval_result(evals_result):
    scores = evals_result['valid_0']['combined_ndcg']
    best_idx = np.argmax(scores)
    epoch = best_idx + 1
    return [epoch,
            evals_result['valid_0']['combined_ndcg'][best_idx],
           evals_result['valid_0']['ndcg_1'][best_idx],
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
    logging.info("setting group for dataset...")
    train_data = set_group_for_dataset(_LOCAL_TRAIN_FILE, query_id_column)
    valid_data = set_group_for_dataset(_LOCAL_VALID_FILE, query_id_column)
    test_data = set_group_for_dataset(_LOCAL_TEST_FILE, query_id_column)
    alpha_values = np.arange(0.5, 1.1, 0.25)
    # alpha_values = [0.9]
    best_eval_result = []
    logging.info("Loading pre-trained model...")
    pretrained_model = lgb.Booster(model_file=f"{_LOCAL_MODEL_PATH}/model")
    for alpha in alpha_values:
        evals_result = {}
        train_data.alpha = alpha
        valid_data.alpha = alpha
        logging.info("Training model...")
        #for more metrics https://github.com/microsoft/LightGBM/blob/a7885b60dd398fc89d797049a31c7b85713c966b/examples/python-guide/advanced_example.py#L175-L185
        model = lgb.train(params=tree_params,
                         train_set=train_data,
                         valid_sets=[valid_data],
                         fobj=customized_objective_freshness,
                         feval=customized_eval_freshness,
                         # callbacks=[lgb.print_evaluation()],
                         evals_result=evals_result,
                         keep_training_booster=True,
                         init_model=pretrained_model,
                         )
        # model.save_model(f"tmp/model_{alpha}.txt")
        # _copy_model_to_gcs(f"tmp/model_{alpha}.txt", args.output_model_path)
        test_data.alpha = alpha
        logging.info("Predicting on test data...")
        y_pred = model.predict(_LOCAL_TEST_FILE)
        logging.info("Report Metrics on test data ...")
        test_results = report_metrics_freshness(y_pred, test_data)
        best_eval_result.append(test_results)

    df = pandas.DataFrame.from_records(best_eval_result, columns=['ndcg_1', 'ndcg_2'])
    df['alpha'] = alpha_values
    print(df)

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
    ap.add_argument("--bz-features-path-test")
    ap.add_argument("--tree-config-path")
    ap.add_argument("--train-ratio", default=0.9)
    ap.add_argument("--pretrained-model-path")
    ap.add_argument("--output-model-path")
    ap.add_argument("--make-validation-ndcg-purchase-only", action="store_true")
    args, extra_args = ap.parse_known_args()

    logging.basicConfig(stream=sys.stderr)

    _create_local_directories()
    _copy_tree_config_locally(args.tree_config_path, _LOCAL_TREE_CONFIG_PATH)
    tree_params = _parse_tree_params(_LOCAL_TREE_CONFIG_PATH, extra_args)

    query_id_column = _extract_query_id_column(tree_params)
    # _copy_bz_features_locally(args.bz_features_path, _LOCAL_BZ_FEATURES_DIR)
    # _copy_bz_features_locally(args.bz_features_path_test, _LOCAL_BZ_FEATURES_DIR_TEST)
    # _create_train_valid_files(args.train_ratio, query_id_column)
    #
    # _copy_model_locally(args.pretrained_model_path, _LOCAL_MODEL_PATH)
    _train_model_report_metrics(tree_params,
                                args.make_validation_ndcg_purchase_only)


