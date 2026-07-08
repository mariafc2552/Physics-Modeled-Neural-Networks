import os
import time
import csv
import pickle
import logging
import shutil
import datetime
import yaml
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from src.data import Reader
from src.auxiliar import longtiming

logger = logging.getLogger(__name__)


IRIS_FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def as_list(value):
    """
    Convert scalar values to one-element lists for grid-search compatibility.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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
        y: shape (num_samples,)

    For Iris:
        X.shape = (N, 4)
        y.shape = (N,)
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

        targets_np = targets_np.reshape(-1)

        X_batches.append(inputs_np.astype(np.float32))
        y_batches.append(targets_np.astype(np.int64))

    if len(X_batches) == 0:
        raise ValueError("The dataset is empty. No data could be loaded.")

    X = np.concatenate(X_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)

    return X, y


def compute_multiclass_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
) -> float:
    """
    Compute macro specificity for multi-class classification.

    For each class, specificity is computed one-vs-rest as:

        TN / (TN + FP)

    The returned value is the macro average across classes.
    """
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    specificities = []

    for class_index in range(len(labels)):
        tp = cm[class_index, class_index]
        fp = cm[:, class_index].sum() - tp
        fn = cm[class_index, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)

    return float(np.mean(specificities))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute multi-class classification metrics for Iris.

    Labels are expected to be integers:
        0, 1, or 2

    Precision, recall, specificity and F1 are computed as macro averages.
    """
    labels = [0, 1, 2]

    accuracy = accuracy_score(y_true, y_pred)
    error = 1.0 - accuracy

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    specificity = compute_multiclass_specificity(
        y_true,
        y_pred,
        labels=labels,
    )

    if y_score is not None and len(np.unique(y_true)) == 3:
        try:
            roc_auc = roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            roc_auc = float("nan")
    else:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "error": float(error),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(
    model: DecisionTreeClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate a fitted DecisionTreeClassifier.
    """
    y_pred = model.predict(X)

    y_score = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if proba.ndim == 2 and proba.shape[1] == 3:
            y_score = proba

    metrics = compute_metrics(
        y_true=y,
        y_pred=y_pred,
        y_score=y_score,
    )

    return metrics


def get_tree_complexity(model: DecisionTreeClassifier) -> Dict[str, int]:
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
    Build a grid of DecisionTreeClassifier hyperparameters from config.yaml.
    """
    criterion_values = as_list(model_config.get("criterion", ["gini"]))
    max_depth_values = as_list(model_config.get("max_depth", [None]))
    min_samples_split_values = as_list(model_config.get("min_samples_split", [2]))
    min_samples_leaf_values = as_list(model_config.get("min_samples_leaf", [1]))
    max_features_values = as_list(model_config.get("max_features", [None]))
    class_weight_values = as_list(model_config.get("class_weight", [None]))
    ccp_alpha_values = as_list(model_config.get("ccp_alpha", [0.0]))

    param_grid = []

    for (
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        class_weight,
        ccp_alpha,
    ) in product(
        criterion_values,
        max_depth_values,
        min_samples_split_values,
        min_samples_leaf_values,
        max_features_values,
        class_weight_values,
        ccp_alpha_values,
    ):
        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": int(min_samples_split),
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
            "class_weight": class_weight,
            "ccp_alpha": float(ccp_alpha),
        }

        param_grid.append(params)

    return param_grid


def make_config_name(params: Dict[str, Any]) -> str:
    """
    Create folder/file name from a hyperparameter dictionary.
    """
    return (
        f"dt_criterion_{tag_value(params['criterion'])}"
        f"_max_depth_{tag_value(params['max_depth'])}"
        f"_min_split_{tag_value(params['min_samples_split'])}"
        f"_min_leaf_{tag_value(params['min_samples_leaf'])}"
        f"_max_features_{tag_value(params['max_features'])}"
        f"_class_weight_{tag_value(params['class_weight'])}"
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
        f.write("Decision Tree configuration\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset:\n")
        f.write("Iris\n")
        f.write("Task: multi-class classification\n")
        f.write("Classes: 0, 1, 2\n\n")

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
        "runs_decision_tree_iris",
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

    if X_train.shape[1] != 4:
        raise ValueError(
            f"Iris requires 4 input features, but X_train has {X_train.shape[1]}."
        )

    if len(np.unique(y_train)) != 3:
        raise ValueError(
            "This script expects Iris multi-class labels with three classes."
        )

    model_config = config.get("model", {})
    random_state = model_config.get("random_state", 42)

    param_grid = build_param_grid(model_config)

    print(f"Number of Decision Tree configurations: {len(param_grid)}")

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
        "class_weight",
        "ccp_alpha",
        "valid_accuracy",
        "valid_error",
        "valid_precision",
        "valid_recall",
        "valid_specificity",
        "valid_f1",
        "valid_roc_auc",
        "test_accuracy",
        "test_error",
        "test_precision",
        "test_recall",
        "test_specificity",
        "test_f1",
        "test_roc_auc",
        "tree_depth",
        "num_leaves",
        "num_nodes",
        "train_time_s",
        "test_time_s",
    ]

    with open(hyperparameter_results_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()

    best_valid_accuracy = -float("inf")
    best_valid_f1 = -float("inf")
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

        model = DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
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

        print(f"Validation accuracy: {valid_metrics['accuracy']:.6f}")
        print(f"Validation error: {valid_metrics['error']:.6f}")
        print(f"Validation F1: {valid_metrics['f1']:.6f}")
        print(f"Tree depth: {complexity['tree_depth']}")
        print(f"Number of leaves: {complexity['num_leaves']}")
        print(f"Test accuracy: {test_metrics['accuracy']:.6f}")
        print(f"Test error: {test_metrics['error']:.6f}")
        print(f"Training time: {train_time_s:.6f}s")
        print(f"Test inference time: {test_time_s:.6f}s")

        model_path = os.path.join(config_results_folder, "decision_tree_model.pkl")
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
            tree_text = export_text(
                model,
                feature_names=IRIS_FEATURE_NAMES,
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
                    "class_weight": params["class_weight"],
                    "ccp_alpha": params["ccp_alpha"],
                    "valid_accuracy": valid_metrics["accuracy"],
                    "valid_error": valid_metrics["error"],
                    "valid_precision": valid_metrics["precision"],
                    "valid_recall": valid_metrics["recall"],
                    "valid_specificity": valid_metrics["specificity"],
                    "valid_f1": valid_metrics["f1"],
                    "valid_roc_auc": valid_metrics["roc_auc"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_error": test_metrics["error"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_specificity": test_metrics["specificity"],
                    "test_f1": test_metrics["f1"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "tree_depth": complexity["tree_depth"],
                    "num_leaves": complexity["num_leaves"],
                    "num_nodes": complexity["num_nodes"],
                    "train_time_s": train_time_s,
                    "test_time_s": test_time_s,
                }
            )

        is_better = False

        if valid_metrics["accuracy"] > best_valid_accuracy:
            is_better = True

        elif valid_metrics["accuracy"] == best_valid_accuracy:
            if valid_metrics["f1"] > best_valid_f1:
                is_better = True

            elif valid_metrics["f1"] == best_valid_f1:
                if complexity["tree_depth"] < best_tree_depth:
                    is_better = True

                elif complexity["tree_depth"] == best_tree_depth:
                    if complexity["num_leaves"] < best_num_leaves:
                        is_better = True

        if is_better:
            best_valid_accuracy = valid_metrics["accuracy"]
            best_valid_f1 = valid_metrics["f1"]
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
            "decision_tree_iris.pkl",
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
            best_tree_text = export_text(
                best_model,
                feature_names=IRIS_FEATURE_NAMES,
            )
            with open(best_tree_text_path, "w") as f:
                f.write(best_tree_text)
        except Exception as exc:
            logger.warning(f"Could not export best tree text: {exc}")

        print("\n" + "=" * 80)
        print("BEST DECISION TREE CONFIGURATION")
        print("=" * 80)
        print(f"Best config name: {best_config_name}")
        print(f"Best params: {best_params}")
        print(f"Best validation accuracy: {best_valid_metrics['accuracy']:.6f}")
        print(f"Best validation error: {best_valid_metrics['error']:.6f}")
        print(f"Best validation F1: {best_valid_metrics['f1']:.6f}")
        print(f"Test accuracy: {best_test_metrics['accuracy']:.6f}")
        print(f"Test error: {best_test_metrics['error']:.6f}")
        print(f"Test F1: {best_test_metrics['f1']:.6f}")
        print(f"Tree depth: {best_complexity['tree_depth']}")
        print(f"Number of leaves: {best_complexity['num_leaves']}")
        print(f"Best model saved to: {os.path.abspath(best_model_path)}")
        print(f"Total grid-search time: {longtiming(elapsed_s)}")

    else:
        print("No valid Decision Tree configuration was found.")

