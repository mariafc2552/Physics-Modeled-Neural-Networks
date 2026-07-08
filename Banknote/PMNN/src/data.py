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
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

np.random.seed(42)

BANKNOTE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00267/data_banknote_authentication.txt"
)

BANKNOTE_FEATURE_COLUMNS = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
]

BANKNOTE_COLUMNS = BANKNOTE_FEATURE_COLUMNS + ["target"]


class Reader:
    """
    Class for reading Banknote Authentication dataset splits from directory.

    Expected structure:

        datadir/
        ├── train/
        │   ├── banknote__b001.npz
        │   ├── banknote__b002.npz
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
        * target classes = 0, 1

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


def find_banknote_file(srcdir: str):
    """
    Search for a local Banknote Authentication file inside srcdir.

    Accepted extensions:
        .txt, .data, .csv

    The function gives priority to filenames containing:
        banknote
    """
    patterns = ["*.txt", "*.data", "*.csv"]

    candidates = []

    for pattern in patterns:
        candidates.extend(glob(os.path.join(srcdir, pattern)))

    if len(candidates) == 0:
        return None

    preferred = [
        f for f in candidates
        if "banknote" in os.path.basename(f).lower()
    ]

    if len(preferred) > 0:
        return sorted(preferred)[0]

    return sorted(candidates)[0]


def read_banknote_file(file_path_or_url: str):
    """
    Read Banknote Authentication data from a local file or from the UCI URL.

    Expected format:
        variance, skewness, curtosis, entropy, target

    The original UCI file has no header. Some CSV versions may include a
    header row; this function safely removes it by converting all columns
    to numeric values and dropping non-numeric rows.
    """
    df_raw = pd.read_csv(file_path_or_url, header=None)

    if df_raw.shape[1] < 5:
        raise ValueError(
            "Banknote Authentication data must contain at least 5 columns: "
            "4 features and 1 target column."
        )

    df = df_raw.iloc[:, :5].copy()
    df.columns = BANKNOTE_COLUMNS

    for column in BANKNOTE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(axis=0).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid numeric rows found in the Banknote dataset.")

    unique_labels = sorted(df["target"].astype(int).unique().tolist())

    if len(unique_labels) != 2:
        raise ValueError(
            "Banknote Authentication is expected to have exactly 2 classes. "
            f"Found labels: {unique_labels}"
        )

    original_label_mapping = {
        int(original_label): int(new_label)
        for new_label, original_label in enumerate(unique_labels)
    }

    df["target"] = (
        df["target"]
        .astype(int)
        .map(original_label_mapping)
        .astype(np.int64)
    )

    return df, original_label_mapping


def load_banknote_dataframe(srcdir: str):
    """
    Load Banknote Authentication dataset.

    Priority:
        1. Use a local file found in srcdir.
        2. If no local file is found, download it directly from UCI.
    """
    local_file = find_banknote_file(srcdir)

    if local_file is not None:
        df, original_label_mapping = read_banknote_file(local_file)
        source = local_file
    else:
        print(
            "No local Banknote file was found in srcdir. "
            "Trying to load the dataset from the UCI URL..."
        )
        df, original_label_mapping = read_banknote_file(BANKNOTE_URL)
        source = BANKNOTE_URL

    return df, source, original_label_mapping


def save_split(split_name, Xs, ys, datadir, batch_size, base_name):
    """
    Save one split as .npz files.

    Unlike the previous version, this function does not discard the last
    incomplete batch. This is important for datasets where the test split
    may contain fewer samples than the selected batch size.
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
    # Load Banknote Authentication dataset
    # -------------------------------------------------------------------------
    # Classification task:
    #   * 4 numerical features
    #   * 2 classes
    #
    # Data format:
    #   variance, skewness, curtosis, entropy, target
    # -------------------------------------------------------------------------

    df_raw, raw_source, original_label_mapping = load_banknote_dataframe(srcdir)

    X_np = df_raw[BANKNOTE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_np = df_raw["target"].to_numpy(dtype=np.int64)

    df = pd.DataFrame(X_np, columns=BANKNOTE_FEATURE_COLUMNS)
    df["target"] = y_np
    df["target_name"] = [f"class_{int(i)}" for i in y_np]

    csv_path = os.path.join(srcdir, "banknote_authentication.csv")
    df.to_csv(csv_path, index=False)

    print(f"Raw dataset source: {raw_source}")
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

    target_names = ["class_0", "class_1"]

    label_mapping = {
        "class_0": 0,
        "class_1": 1,
    }

    metadata = {
        "dataset": "Banknote Authentication",
        "source": str(raw_source),
        "task": "binary classification",
        "num_samples": int(X_np.shape[0]),
        "num_features": int(X_np.shape[1]),
        "feature_names": [str(name) for name in BANKNOTE_FEATURE_COLUMNS],
        "target_names": target_names,
        "num_classes": int(len(target_names)),
        "label_mapping": label_mapping,
        "original_label_mapping": {
            int(original): int(encoded)
            for original, encoded in original_label_mapping.items()
        },
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
    base_name = "banknote"

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

