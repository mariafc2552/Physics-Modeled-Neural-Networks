import os
import shutil
import logging
from copy import deepcopy
from glob import glob
from typing import List

import yaml
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

np.random.seed(42)


class Reader:
    """
    Class for reading Iris dataset splits from directory.

    Expected structure:

        datadir/
        ├── train/
        │   ├── iris__b001.npz
        │   ├── iris__b002.npz
        │   └── ...
        ├── valid/
        │   └── ...
        └── test/
            └── ...

    Each .npz contains:
        * features: array of shape (batch_size, n_features)
        * target:   array of shape (batch_size,)

    For this dataset:
        * n_features = 4
        * target classes = 0, 1, 2

    The targets are integer class labels, directly compatible with
    torch.nn.CrossEntropyLoss.
    """

    def __init__(self, folder):
        self.keys = ["train", "valid", "test"]
        self.folder = folder

        self.files = {
            k: sorted(glob(os.path.join(self.folder, k, "*.npz")))
            for k in self.keys
        }

    @staticmethod
    def read(f):
        """
        Read single .npz file to tensors.

        Inputs:
            * f: path to .npz file.

        Outputs:
            * features: torch.FloatTensor, shape (batch_size, 4)
            * target: torch.LongTensor, shape (batch_size,)
        """
        data = dict(np.load(f))

        features = torch.tensor(data["features"], dtype=torch.float32)
        target = torch.tensor(data["target"], dtype=torch.long).view(-1)

        return features, target

    def __call__(self, key=None):
        if key is not None:
            if key not in self.keys:
                raise ValueError(f"Invalid key '{key}'. Must be one of {self.keys}")
            return self.files[key]

        return self.files

    def make_dataset(self, key: str) -> TensorDataset:
        files = self(key)

        feats: List[torch.Tensor] = []
        targs: List[torch.Tensor] = []

        for f in files:
            x, y = Reader.read(f)
            feats.append(x)
            targs.append(y)

        if not feats or not targs:
            raise ValueError(f"No valid data found for key '{key}'")

        X = torch.cat(feats, dim=0)
        y = torch.cat(targs, dim=0)

        return TensorDataset(X, y)

    def make_loaders(self, batch_size: int = None):
        bsz = int(batch_size)

        train_dataset = self.make_dataset("train")
        val_dataset = self.make_dataset("valid")
        test_dataset = self.make_dataset("test")

        train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bsz, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)

        return train_loader, val_loader, test_loader


class Scaler(nn.Module):
    """
    Scaler class to implement feature standardization or min-max normalization.

    The scaler is fitted only on the training set and then applied to train,
    validation and test sets. This avoids data leakage.
    """

    def __init__(self, norm="std", *, dim=-1):
        super().__init__()

        self.norm = norm
        assert self.norm in ["std", "minmax"]

        self.register_buffer("dim", torch.tensor(dim))

        self.register_buffer("min", None)
        self.register_buffer("max", None)

        self.register_buffer("mean", None)
        self.register_buffer("std", None)

        self.cumlen = torch.tensor(0, dtype=torch.int64)

    def reset(self):
        self.register_buffer("min", None)
        self.register_buffer("max", None)

        self.register_buffer("mean", None)
        self.register_buffer("std", None)

        self.cumlen = torch.tensor(0, dtype=torch.int64)

    @torch.no_grad()
    def fit(self, x):
        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            self.min = x_.min(dim=0, keepdim=True)[0]
            self.max = x_.max(dim=0, keepdim=True)[0]

        elif self.norm == "std":
            self.mean = x_.mean(dim=0, keepdim=True)[0]
            self.std = x_.std(dim=0, keepdim=True)[0]

        else:
            raise NotImplementedError

    @torch.no_grad()
    def transform(self, x):
        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            assert self.min is not None and self.max is not None, "Scaler is not fitted!"
            x_ = (x_ - self.min) / (self.max - self.min + 1e-12)

        elif self.norm == "std":
            assert self.mean is not None and self.std is not None, "Scaler is not fitted!"
            x_ = (x_ - self.mean) / (self.std + 1e-12)

        else:
            raise NotImplementedError

        return x_.reshape(x.transpose(-1, self.dim.item()).shape).transpose(
            -1,
            self.dim.item(),
        )

    @torch.no_grad()
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    @torch.no_grad()
    def inverse_transform(self, x):
        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            assert self.min is not None and self.max is not None, "Scaler is not fitted!"
            x_ = x_ * (self.max - self.min + 1e-12) + self.min

        elif self.norm == "std":
            assert self.mean is not None and self.std is not None, "Scaler is not fitted!"
            x_ = x_ * (self.std + 1e-12) + self.mean

        else:
            raise NotImplementedError

        return x_.reshape(x.transpose(-1, self.dim.item()).shape).transpose(
            -1,
            self.dim.item(),
        )

    @torch.no_grad()
    def forward(self, x):
        return self.transform(x)

    def save(self, filename):
        torch.save(deepcopy(self.state_dict()), filename)

    def load(self, filename):
        sd = torch.load(filename, weights_only=True)

        is_minmax = {"min", "max"}.issubset(sd.keys())
        is_std = {"mean", "std"}.issubset(sd.keys())

        self.dim = sd["dim"]

        assert not (is_minmax and is_std)

        if is_minmax:
            self.min, self.max = sd["min"], sd["max"]
            self.norm = "minmax"

        elif is_std:
            self.mean, self.std = sd["mean"], sd["std"]
            self.norm = "std"


def get_split_ratios(config):
    """
    Read split ratios from config.

    Supports the preferred format:

        data:
          splits:
            train: 0.7
            valid: 0.2
            test: 0.1

    and also the older format:

        data:
          train: 0.7
          valid: 0.2
          test: 0.1
    """
    data_cfg = config.get("data", {})

    if "splits" in data_cfg:
        splits = data_cfg["splits"]
    else:
        splits = {
            "train": data_cfg.get("train", 0.7),
            "valid": data_cfg.get("valid", 0.2),
            "test": data_cfg.get("test", 0.1),
        }

    train_ratio = float(splits.get("train", 0.7))
    valid_ratio = float(splits.get("valid", 0.2))
    test_ratio = float(splits.get("test", 0.1))

    ratio_sum = train_ratio + valid_ratio + test_ratio

    if ratio_sum <= 0.0:
        raise ValueError("The sum of train, valid and test ratios must be positive.")

    train_ratio /= ratio_sum
    valid_ratio /= ratio_sum
    test_ratio /= ratio_sum

    return train_ratio, valid_ratio, test_ratio


def save_split(split_name, Xs, ys, datadir, batch_size, base_name):
    """
    Save one split as .npz files.

    Unlike the previous version, this function does not discard the last
    incomplete batch. This is important for small datasets such as Iris,
    where the test split may contain fewer samples than the selected batch size.
    """
    count_total = Xs.shape[0]

    if count_total == 0:
        raise ValueError(f"The split '{split_name}' contains 0 samples.")

    split_dir = os.path.join(datadir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    if batch_size is None:
        out_path = os.path.join(split_dir, f"{base_name}.npz")

        np.savez(
            file=out_path,
            features=Xs.numpy().astype(np.float32),
            target=ys.numpy().astype(np.int64),
        )

        saved = count_total

    else:
        bsz = int(batch_size)

        if bsz <= 0:
            raise ValueError(f"batch_size must be positive, got {bsz}.")

        num_batches = int(np.ceil(count_total / bsz))
        saved = 0

        for b in range(num_batches):
            start = b * bsz
            end = min((b + 1) * bsz, count_total)

            xs = Xs[start:end]
            ys_ = ys[start:end]

            out_path = os.path.join(
                split_dir,
                f"{base_name}__b{b + 1:03d}.npz",
            )

            np.savez(
                file=out_path,
                features=xs.numpy().astype(np.float32),
                target=ys_.numpy().astype(np.int64),
            )

            saved += xs.shape[0]

    discarded = count_total - saved

    logger.info(
        f"{split_name.upper()}: {count_total} samples initially; "
        f"{saved} saved to disk; discarded={discarded}"
    )

    print(
        f"{split_name.upper()}: {count_total} samples initially; "
        f"{saved} saved; discarded={discarded}"
    )

    return saved




if __name__ == "__main__":

    rootdir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

    config_path = os.path.join(rootdir, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"The configuration file is empty or invalid: {config_path}")

    srcdir = os.path.abspath(os.path.join(rootdir, config["paths"]["srcdir"]))
    datadir = os.path.abspath(os.path.join(rootdir, config["paths"]["datadir"]))

    os.makedirs(srcdir, exist_ok=True)

    train_ratio, valid_ratio, test_ratio = get_split_ratios(config)

    if os.path.exists(datadir):
        shutil.rmtree(datadir)
        remove_existing = True
    else:
        remove_existing = False

    os.makedirs(datadir, exist_ok=True)

    for split_name in ["train", "valid", "test"]:
        os.makedirs(os.path.join(datadir, split_name), exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(datadir, "datainfo.log"),
        level=logging.INFO,
    )

    if remove_existing:
        logger.warning(f"Removing existing directory: {datadir}")

    # -------------------------------------------------------------------------
    # Load Iris dataset
    # -------------------------------------------------------------------------
    # Classification task:
    #   * 4 numerical features
    #   * 3 classes
    #
    # In sklearn:
    #   * target = 0 -> setosa
    #   * target = 1 -> versicolor
    #   * target = 2 -> virginica
    # -------------------------------------------------------------------------

    dataset = load_iris()

    X_np = dataset.data.astype(np.float32)
    y_np = dataset.target.astype(np.int64)

    df = pd.DataFrame(X_np, columns=dataset.feature_names)
    df["target"] = y_np
    df["target_name"] = [dataset.target_names[i] for i in y_np]

    csv_path = os.path.join(srcdir, "iris.csv")
    df.to_csv(csv_path, index=False)

    print(f"Raw dataset saved to: {csv_path}")

    # -------------------------------------------------------------------------
    # Stratified train / validation / test split
    # -------------------------------------------------------------------------

    if test_ratio > 0.0:
        X_train_valid, X_test, y_train_valid, y_test = train_test_split(
            X_np,
            y_np,
            test_size=test_ratio,
            random_state=42,
            stratify=y_np,
        )
    else:
        X_train_valid = X_np
        y_train_valid = y_np
        X_test = np.empty((0, X_np.shape[1]), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.int64)

    if valid_ratio > 0.0:
        relative_valid_ratio = valid_ratio / (train_ratio + valid_ratio)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_valid,
            y_train_valid,
            test_size=relative_valid_ratio,
            random_state=42,
            stratify=y_train_valid,
        )
    else:
        X_train = X_train_valid
        y_train = y_train_valid
        X_valid = np.empty((0, X_np.shape[1]), dtype=np.float32)
        y_valid = np.empty((0,), dtype=np.int64)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # -------------------------------------------------------------------------
    # Feature scaling
    # -------------------------------------------------------------------------
    # The scaler is fitted only on the training split and then applied to
    # train, validation and test. This avoids data leakage.
    # -------------------------------------------------------------------------

    scaler_cfg = config.get("data", {}).get(
        "scaler",
        {
            "norm": "std",
            "dim": -1,
        },
    )

    scaler = Scaler(
        norm=scaler_cfg.get("norm", "std"),
        dim=scaler_cfg.get("dim", -1),
    )

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    scaler_path = os.path.join(datadir, "scaler.pt")
    scaler.save(scaler_path)

    print(f"Scaler saved to: {scaler_path}")

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------

    label_mapping = {
        str(name): int(index)
        for index, name in enumerate(dataset.target_names)
    }

    metadata = {
        "dataset": "Iris",
        "source": "sklearn.datasets.load_iris",
        "task": "multiclass classification",
        "num_samples": int(X_np.shape[0]),
        "num_features": int(X_np.shape[1]),
        "feature_names": [str(name) for name in dataset.feature_names],
        "target_names": [str(name) for name in dataset.target_names],
        "num_classes": int(len(dataset.target_names)),
        "label_mapping": label_mapping,
        "split_ratios": {
            "train": float(train_ratio),
            "valid": float(valid_ratio),
            "test": float(test_ratio),
        },
        "train_size": int(X_train.shape[0]),
        "valid_size": int(X_valid.shape[0]),
        "test_size": int(X_test.shape[0]),
        "scaler": {
            "norm": scaler.norm,
            "dim": int(scaler.dim.item()),
        },
    }

    metadata_path = os.path.join(datadir, "metadata.yaml")

    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    print(f"Metadata saved to: {metadata_path}")

    # -------------------------------------------------------------------------
    # Save split batches
    # -------------------------------------------------------------------------

    batch_size = config["data"].get("batch_size", None)
    base_name = "iris"

    saved_train = save_split(
        "train",
        X_train,
        y_train,
        datadir,
        batch_size,
        base_name,
    )

    saved_valid = save_split(
        "valid",
        X_valid,
        y_valid,
        datadir,
        batch_size,
        base_name,
    )

    saved_test = save_split(
        "test",
        X_test,
        y_test,
        datadir,
        batch_size,
        base_name,
    )

    logger.info(
        "Final summary: "
        f"TRAIN={saved_train}, VALID={saved_valid}, TEST={saved_test}"
    )

    print(
        "Final summary ->",
        f"TRAIN={saved_train}, VALID={saved_valid}, TEST={saved_test}",
    )

    print("\nData creation completed successfully.")
    print(f"Output directory: {datadir}")

