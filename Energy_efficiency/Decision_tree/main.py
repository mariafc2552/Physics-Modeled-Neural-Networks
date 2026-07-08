import os
import time
import csv
import pickle
import logging
import shutil
import datetime
import yaml
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.data import Reader
from src.auxiliar import longtiming

logger = logging.getLogger(__name__)


ENERGY_FEATURE_NAMES = [
    "relative_compactness",
    "surface_area",
    "wall_area",
    "roof_area",
    "overall_height",
    "orientation",
    "glazing_area",
    "glazing_area_distribution",
]

ENERGY_TARGET_NAMES = [
    "heating_load",
    "cooling_load",
]


def as_list(value):
    """
    Convert scalar values to one-element lists for grid-search compatibility.
    """
    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def normalize_none(value):
    """
    Convert string representations of null values to Python None.
    """
    if isinstance(value, str):
        if value.lower() in ["none", "null"]:
            return None

    return value


def tag_value(value):
    """
    Create a safe string tag for folder/file names.
    """
    if value is None:
        return "None"

    return str(value).replace(".", "p").replace("/", "_").replace("\\", "_")


def dataset_to_numpy(dataset, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a PyTorch Dataset returned by Reader.make_dataset into NumPy arrays.

    Expected dataset item:
        features, target

    Output:
        X: shape (num_samples, num_features)
        y: shape (num_samples, num_targets)

    For Energy Efficiency:
        X.shape = (N, 8)
        y.shape = (N, 2)

    The two target variables are:
        y[:, 0] -> heating_load
        y[:, 1] -> cooling_load
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    X_batches = []
    y_batches = []

    for inputs, targets in loader:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu()

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu()

        inputs_np = np.asarray(inputs)
        targets_np = np.asarray(targets)

        if inputs_np.ndim > 2:
            inputs_np = inputs_np.reshape(inputs_np.shape[0], -1)

        if targets_np.ndim == 1:
            targets_np = targets_np.reshape(-1, 1)

        elif targets_np.ndim > 2:
            targets_np = targets_np.reshape(targets_np.shape[0], -1)

        X_batches.append(inputs_np.astype(np.float32))
        y_batches.append(targets_np.astype(np.float32))

    if len(X_batches) == 0:
        raise ValueError("The dataset is empty. No data could be loaded.")

    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)

    return X, y


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None,
) -> Dict[str, float]:
    """
    Compute multi-output regression metrics.

    Metrics:
        MSE  : Mean Squared Error
        RMSE : Root Mean Squared Error
        MAE  : Mean Absolute Error
        R2   : Coefficient of determination

    For Energy Efficiency, target-specific metrics are also computed for:
        * heating_load
        * cooling_load
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch in metrics: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)

    try:
        r2 = r2_score(y_true, y_pred)
    except ValueError:
        r2 = float("nan")

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    if target_names is not None and y_true.shape[1] == len(target_names):
        raw_mse = np.atleast_1d(
            mean_squared_error(y_true, y_pred, multioutput="raw_values")
        )
        raw_mae = np.atleast_1d(
            mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        )

        try:
            raw_r2 = np.atleast_1d(
                r2_score(y_true, y_pred, multioutput="raw_values")
            )
        except ValueError:
            raw_r2 = np.full(y_true.shape[1], np.nan)

        for idx, target_name in enumerate(target_names):
            metrics[f"{target_name}_mse"] = float(raw_mse[idx])
            metrics[f"{target_name}_rmse"] = float(np.sqrt(raw_mse[idx]))
            metrics[f"{target_name}_mae"] = float(raw_mae[idx])
            metrics[f"{target_name}_r2"] = float(raw_r2[idx])

    return metrics


def evaluate_model(
    model: DecisionTreeRegressor,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate a fitted DecisionTreeRegressor.
    """
    y_pred = model.predict(X)

    metrics = compute_regression_metrics(
        y_true=y,
        y_pred=y_pred,
        target_names=ENERGY_TARGET_NAMES,
    )

    return metrics


def get_tree_complexity(model: DecisionTreeRegressor) -> Dict[str, int]:
    """
    Return basic tree-complexity information.
    """
    return {
        "tree_depth": int(model.get_depth()),
        "num_leaves": int(model.get_n_leaves()),
        "num_nodes": int(model.tree_.node_count),
    }


def save_pickle(obj: Any, file_path: str):
    """
    Save Python object as pickle.
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def build_param_grid(model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a grid of DecisionTreeRegressor hyperparameters from config.yaml.

    Expected regression config:

        model:
          criterion: [squared_error, friedman_mse, absolute_error]
          max_depth: [null, 2, 3, 4, 5, 10]
          min_samples_split: [2, 5, 10]
          min_samples_leaf: [1, 2, 5, 10]
          max_features: [null, sqrt, log2]
          ccp_alpha: [0.0, 0.001, 0.01]
          random_state: 42
    """
    criterion_values = as_list(model_config.get("criterion", ["squared_error"]))
    max_depth_values = as_list(model_config.get("max_depth", [None]))
    min_samples_split_values = as_list(model_config.get("min_samples_split", [2]))
    min_samples_leaf_values = as_list(model_config.get("min_samples_leaf", [1]))
    max_features_values = as_list(model_config.get("max_features", [None]))
    ccp_alpha_values = as_list(model_config.get("ccp_alpha", [0.0]))

    param_grid = []

    for (
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        ccp_alpha,
    ) in product(
        criterion_values,
        max_depth_values,
        min_samples_split_values,
        min_samples_leaf_values,
        max_features_values,
        ccp_alpha_values,
    ):
        max_depth = normalize_none(max_depth)
        max_features = normalize_none(max_features)

        params = {
            "criterion": criterion,
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_split": int(min_samples_split),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
            "ccp_alpha": float(ccp_alpha),
        }

        param_grid.append(params)

    return param_grid


def make_config_name(params: Dict[str, Any]) -> str:
    """
    Create folder/file name from a hyperparameter dictionary.
    """
    return (
        f"dtr_criterion_{tag_value(params['criterion'])}"
        f"_max_depth_{tag_value(params['max_depth'])}"
        f"_min_split_{tag_value(params['min_samples_split'])}"
        f"_min_leaf_{tag_value(params['min_samples_leaf'])}"
        f"_max_features_{tag_value(params['max_features'])}"
        f"_ccp_alpha_{tag_value(params['ccp_alpha'])}"
    )


def write_metrics_report(
    file_path: str,
    *,
    params: Dict[str, Any],
    train_metrics: Dict[str, Any],
    valid_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    complexity: Dict[str, int],
    train_time_s: float,
    test_time_s: float,
):
    """
    Save a readable report for one model configuration.
    """
    with open(file_path, "w") as f:
        f.write("Decision Tree Regressor configuration\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset:\n")
        f.write("Energy Efficiency\n")
        f.write("Task: multi-output regression\n")
        f.write("Targets: heating_load, cooling_load\n\n")

        f.write("Hyperparameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTree complexity:\n")
        for key, value in complexity.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTrain metrics:\n")
        for key, value in train_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nValidation metrics:\n")
        for key, value in valid_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTest metrics:\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTiming:\n")
        f.write(f"train_time_s: {train_time_s}\n")
        f.write(f"test_time_s: {test_time_s}\n")
        f.write(f"train_time_hms: {longtiming(train_time_s)}\n")
        f.write(f"test_time_hms: {longtiming(test_time_s)}\n")


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))

    with open(os.path.join(rootdir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    results_base = config["paths"].get(
        "results",
        "runs_decision_tree_energy_efficiency",
    )

    baserundir = os.path.join(rootdir, results_base)
    os.makedirs(baserundir, exist_ok=True)

    now = datetime.datetime.now()
    rundir = os.path.join(baserundir, f"{now.strftime('%Y%m%d-%H%M')}")

    if os.path.exists(rundir):
        remove_existing = True
        shutil.rmtree(rundir)
    else:
        remove_existing = False

    os.makedirs(rundir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(rundir, "runinfo.log"),
        level=logging.INFO,
    )

    if remove_existing:
        logger.warning(f"Removing existing directory: {os.path.abspath(rundir)}")

    logger.info(f"Run directory: {os.path.abspath(rundir)}")

    print(f"Run directory: {os.path.abspath(rundir)}")

    datadir = config["paths"]["datadir"]
    reader = Reader(datadir)

    train_dataset = reader.make_dataset("train")
    valid_dataset = reader.make_dataset("valid")
    test_dataset = reader.make_dataset("test")

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_valid, y_valid = dataset_to_numpy(valid_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Valid shape: X={X_valid.shape}, y={y_valid.shape}")
    print(f"Test shape:  X={X_test.shape}, y={y_test.shape}")

    if X_train.ndim != 2:
        raise ValueError(
            f"Expected X_train with shape (num_samples, num_features), "
            f"but received {X_train.shape}."
        )

    if X_train.shape[1] != 8:
        print(
            f"Warning: Energy Efficiency normally has 8 features, "
            f"but X_train has {X_train.shape[1]} features."
        )

    if y_train.ndim != 2 or y_train.shape[1] != 2:
        raise ValueError(
            "Energy Efficiency requires a multi-output target with shape "
            f"(num_samples, 2), but received y_train.shape={y_train.shape}."
        )

    if y_valid.ndim != 2 or y_valid.shape[1] != 2:
        raise ValueError(
            "Energy Efficiency requires a multi-output target with shape "
            f"(num_samples, 2), but received y_valid.shape={y_valid.shape}."
        )

    if y_test.ndim != 2 or y_test.shape[1] != 2:
        raise ValueError(
            "Energy Efficiency requires a multi-output target with shape "
            f"(num_samples, 2), but received y_test.shape={y_test.shape}."
        )

    if not np.issubdtype(y_train.dtype, np.number):
        raise ValueError("This script expects numerical continuous targets.")

    model_config = config.get("model", {})
    random_state = model_config.get("random_state", 42)

    if "class_weight" in model_config:
        print(
            "Warning: class_weight was found in config.yaml but will be ignored "
            "because DecisionTreeRegressor does not use class_weight."
        )

    param_grid = build_param_grid(model_config)

    print(f"Number of Decision Tree Regressor configurations: {len(param_grid)}")

    hyperparameter_results_path = os.path.join(
        rundir,
        "hyperparameter_results.csv",
    )

    header = [
        "config_name",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "ccp_alpha",
        "valid_mse",
        "valid_rmse",
        "valid_mae",
        "valid_r2",
        "valid_heating_load_mse",
        "valid_heating_load_rmse",
        "valid_heating_load_mae",
        "valid_heating_load_r2",
        "valid_cooling_load_mse",
        "valid_cooling_load_rmse",
        "valid_cooling_load_mae",
        "valid_cooling_load_r2",
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_heating_load_mse",
        "test_heating_load_rmse",
        "test_heating_load_mae",
        "test_heating_load_r2",
        "test_cooling_load_mse",
        "test_cooling_load_rmse",
        "test_cooling_load_mae",
        "test_cooling_load_r2",
        "tree_depth",
        "num_leaves",
        "num_nodes",
        "train_time_s",
        "test_time_s",
    ]

    with open(hyperparameter_results_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()

    best_valid_mse = float("inf")
    best_valid_mae = float("inf")
    best_tree_depth = float("inf")
    best_num_leaves = float("inf")

    best_params = None
    best_model = None
    best_config_name = None
    best_train_metrics = None
    best_valid_metrics = None
    best_test_metrics = None
    best_complexity = None
    best_train_time_s = None
    best_test_time_s = None

    global_start = time.time()

    for idx, params in enumerate(param_grid, start=1):
        config_name = make_config_name(params)

        print("\n" + "=" * 80)
        print(f"Configuration {idx} of {len(param_grid)}")
        print(config_name)
        print("=" * 80)

        config_results_folder = os.path.join(rundir, config_name)
        os.makedirs(config_results_folder, exist_ok=True)

        model = DecisionTreeRegressor(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            ccp_alpha=params["ccp_alpha"],
            random_state=random_state,
        )

        train_start = time.time()
        model.fit(X_train, y_train)
        train_time_s = time.time() - train_start

        train_metrics = evaluate_model(model, X_train, y_train)
        valid_metrics = evaluate_model(model, X_valid, y_valid)

        test_start = time.time()
        test_metrics = evaluate_model(model, X_test, y_test)
        test_time_s = time.time() - test_start

        complexity = get_tree_complexity(model)

        print(f"Validation MSE: {valid_metrics['mse']:.6f}")
        print(f"Validation RMSE: {valid_metrics['rmse']:.6f}")
        print(f"Validation MAE: {valid_metrics['mae']:.6f}")
        print(f"Validation R2: {valid_metrics['r2']:.6f}")
        print(f"Validation heating_load RMSE: {valid_metrics['heating_load_rmse']:.6f}")
        print(f"Validation cooling_load RMSE: {valid_metrics['cooling_load_rmse']:.6f}")
        print(f"Tree depth: {complexity['tree_depth']}")
        print(f"Number of leaves: {complexity['num_leaves']}")
        print(f"Test MSE: {test_metrics['mse']:.6f}")
        print(f"Test RMSE: {test_metrics['rmse']:.6f}")
        print(f"Test MAE: {test_metrics['mae']:.6f}")
        print(f"Test R2: {test_metrics['r2']:.6f}")
        print(f"Test heating_load RMSE: {test_metrics['heating_load_rmse']:.6f}")
        print(f"Test cooling_load RMSE: {test_metrics['cooling_load_rmse']:.6f}")
        print(f"Training time: {train_time_s:.6f}s")
        print(f"Test inference time: {test_time_s:.6f}s")

        model_path = os.path.join(config_results_folder, "decision_tree_regressor.pkl")
        save_pickle(model, model_path)

        report_path = os.path.join(config_results_folder, "metrics_report.txt")
        write_metrics_report(
            report_path,
            params=params,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            complexity=complexity,
            train_time_s=train_time_s,
            test_time_s=test_time_s,
        )

        tree_text_path = os.path.join(config_results_folder, "tree_structure.txt")

        try:
            feature_names = ENERGY_FEATURE_NAMES

            tree_text = export_text(
                model,
                feature_names=feature_names,
            )

            with open(tree_text_path, "w") as f:
                f.write(tree_text)

        except Exception as exc:
            logger.warning(f"Could not export tree text for {config_name}: {exc}")

        with open(hyperparameter_results_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)

            writer.writerow(
                {
                    "config_name": config_name,
                    "criterion": params["criterion"],
                    "max_depth": params["max_depth"],
                    "min_samples_split": params["min_samples_split"],
                    "min_samples_leaf": params["min_samples_leaf"],
                    "max_features": params["max_features"],
                    "ccp_alpha": params["ccp_alpha"],
                    "valid_mse": valid_metrics["mse"],
                    "valid_rmse": valid_metrics["rmse"],
                    "valid_mae": valid_metrics["mae"],
                    "valid_r2": valid_metrics["r2"],
                    "valid_heating_load_mse": valid_metrics["heating_load_mse"],
                    "valid_heating_load_rmse": valid_metrics["heating_load_rmse"],
                    "valid_heating_load_mae": valid_metrics["heating_load_mae"],
                    "valid_heating_load_r2": valid_metrics["heating_load_r2"],
                    "valid_cooling_load_mse": valid_metrics["cooling_load_mse"],
                    "valid_cooling_load_rmse": valid_metrics["cooling_load_rmse"],
                    "valid_cooling_load_mae": valid_metrics["cooling_load_mae"],
                    "valid_cooling_load_r2": valid_metrics["cooling_load_r2"],
                    "test_mse": test_metrics["mse"],
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                    "test_heating_load_mse": test_metrics["heating_load_mse"],
                    "test_heating_load_rmse": test_metrics["heating_load_rmse"],
                    "test_heating_load_mae": test_metrics["heating_load_mae"],
                    "test_heating_load_r2": test_metrics["heating_load_r2"],
                    "test_cooling_load_mse": test_metrics["cooling_load_mse"],
                    "test_cooling_load_rmse": test_metrics["cooling_load_rmse"],
                    "test_cooling_load_mae": test_metrics["cooling_load_mae"],
                    "test_cooling_load_r2": test_metrics["cooling_load_r2"],
                    "tree_depth": complexity["tree_depth"],
                    "num_leaves": complexity["num_leaves"],
                    "num_nodes": complexity["num_nodes"],
                    "train_time_s": train_time_s,
                    "test_time_s": test_time_s,
                }
            )

        is_better = False

        if valid_metrics["mse"] < best_valid_mse:
            is_better = True

        elif valid_metrics["mse"] == best_valid_mse:
            if valid_metrics["mae"] < best_valid_mae:
                is_better = True

            elif valid_metrics["mae"] == best_valid_mae:
                if complexity["tree_depth"] < best_tree_depth:
                    is_better = True

                elif complexity["tree_depth"] == best_tree_depth:
                    if complexity["num_leaves"] < best_num_leaves:
                        is_better = True

        if is_better:
            best_valid_mse = valid_metrics["mse"]
            best_valid_mae = valid_metrics["mae"]
            best_tree_depth = complexity["tree_depth"]
            best_num_leaves = complexity["num_leaves"]

            best_params = params
            best_model = model
            best_config_name = config_name
            best_train_metrics = train_metrics
            best_valid_metrics = valid_metrics
            best_test_metrics = test_metrics
            best_complexity = complexity
            best_train_time_s = train_time_s
            best_test_time_s = test_time_s

    elapsed_s = time.time() - global_start

    if best_model is not None:
        best_folder = os.path.join(rundir, "best_hparams")

        if os.path.exists(best_folder):
            shutil.rmtree(best_folder)

        os.makedirs(best_folder, exist_ok=True)

        best_model_filename = config["paths"].get(
            "model",
            "decision_tree_energy_efficiency.pkl",
        )

        best_model_path = os.path.join(best_folder, best_model_filename)
        save_pickle(best_model, best_model_path)

        best_summary_path = os.path.join(best_folder, "best_summary.txt")

        write_metrics_report(
            best_summary_path,
            params=best_params,
            train_metrics=best_train_metrics,
            valid_metrics=best_valid_metrics,
            test_metrics=best_test_metrics,
            complexity=best_complexity,
            train_time_s=best_train_time_s,
            test_time_s=best_test_time_s,
        )

        best_tree_text_path = os.path.join(best_folder, "best_tree_structure.txt")

        try:
            feature_names = ENERGY_FEATURE_NAMES

            best_tree_text = export_text(
                best_model,
                feature_names=feature_names,
            )

            with open(best_tree_text_path, "w") as f:
                f.write(best_tree_text)

        except Exception as exc:
            logger.warning(f"Could not export best tree text: {exc}")

        print("\n" + "=" * 80)
        print("BEST DECISION TREE REGRESSOR CONFIGURATION")
        print("=" * 80)
        print(f"Best config name: {best_config_name}")
        print(f"Best params: {best_params}")
        print(f"Best validation MSE: {best_valid_metrics['mse']:.6f}")
        print(f"Best validation RMSE: {best_valid_metrics['rmse']:.6f}")
        print(f"Best validation MAE: {best_valid_metrics['mae']:.6f}")
        print(f"Best validation R2: {best_valid_metrics['r2']:.6f}")
        print(f"Best validation heating_load RMSE: {best_valid_metrics['heating_load_rmse']:.6f}")
        print(f"Best validation cooling_load RMSE: {best_valid_metrics['cooling_load_rmse']:.6f}")
        print(f"Test MSE: {best_test_metrics['mse']:.6f}")
        print(f"Test RMSE: {best_test_metrics['rmse']:.6f}")
        print(f"Test MAE: {best_test_metrics['mae']:.6f}")
        print(f"Test R2: {best_test_metrics['r2']:.6f}")
        print(f"Test heating_load RMSE: {best_test_metrics['heating_load_rmse']:.6f}")
        print(f"Test cooling_load RMSE: {best_test_metrics['cooling_load_rmse']:.6f}")
        print(f"Tree depth: {best_complexity['tree_depth']}")
        print(f"Number of leaves: {best_complexity['num_leaves']}")
        print(f"Best model saved to: {os.path.abspath(best_model_path)}")
        print(f"Total grid-search time: {longtiming(elapsed_s)}")

    else:
        print("No valid Decision Tree Regressor configuration was found.")

