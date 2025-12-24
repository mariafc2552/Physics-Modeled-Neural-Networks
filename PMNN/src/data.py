import shutil
import torch
import logging
import yaml
from sklearn.datasets import fetch_california_housing
from torch import nn
import numpy as np
import os
from copy import deepcopy
import pandas as pd
from glob import glob
from typing import List
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

np.random.seed(42)

class Reader:
    """
    Class for reading California Housing dataset splits from directory.
    """

    def __init__(self, folder):
        """
        Initializer for Reader class.

        Inputs:
            * folder: base directory containing split subdirectories.

        Expected structure:

            .
            └── folder/
                ├── train/
                │   ├── california_housing__b001.npz
                │   ├── california_housing__b002.npz
                │   └── ...
                ├── valid/
                │   └── ...
                └── test/
                    └── ...

            Each .npz contains:
                'features': array of shape (batch_size, n_features)
                'target':   array of shape (batch_size,)
        """
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
            * f: path to .npz file

        Outputs:
            * features: torch.FloatTensor
            * target:   torch.FloatTensor
        """
        data = dict(np.load(f))
        features = torch.tensor(data["features"], dtype=torch.float32)
        target = torch.tensor(data["target"], dtype=torch.float32)
        return features, target

    def __call__(self, key=None):
        """
        Get file list for given split key.

        Inputs:
            * key: 'train', 'valid' or 'test'. If None, returns dict of all splits.

        Outputs:
            * list of files (if key is given) or dict of lists (if key is None)
        """
        if key is not None:
            if key not in self.keys:
                raise ValueError(f"Invalid key '{key}'. Must be one of {self.keys}")
            return self.files[key]
        else:
            return self.files

    def make_dataset(self, key: str) -> TensorDataset:
        """
        Build a TensorDataset from all .npz files of a given split.

        Inputs:
            * key: 'train', 'valid' or 'test'

        Outputs:
            * TensorDataset containing concatenated features and targets
        """
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
        """
        Build DataLoaders for train, valid and test splits.

        Inputs:
            * batch_size: size of the mini-batches (if None, use default_bs)

        Outputs:
            * train_loader, val_loader, test_loader
        """
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
    Scaler class (instance of `torch.nn.Module`) to implement standardization or normalization.
    """

    def __init__(self, norm="std", *, dim=-1):
        """
        Initializer for `Scaler`.

        Inputs:
            * norm: whether to use standardization ('std') or min-max normalization ('minmax').
            * dim: feature dimension. If data has shape (...(N), length, channels), then `dim=-1`.
        """
        super().__init__()

        self.norm = norm
        assert self.norm in ["std", "minmax"]

        self.register_buffer("dim", torch.tensor(dim))

        # minmax
        self.register_buffer("min", None)
        self.register_buffer("max", None)
        # std
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.cumlen = torch.tensor(0, dtype=torch.int64)

    def reset(self):
        """
        Reset scaler to defaults (untrained).
        """
        # minmax
        self.register_buffer("min", None)
        self.register_buffer("max", None)
        # std
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.cumlen = torch.tensor(0, dtype=torch.int64)

    @torch.no_grad()
    def fit(self, x):
        """
        Fit scaler in single step.

        Inputs:
            * x: tensor to fit to.
        """

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
        """
        Transform tensor with trained scaler.

        Inputs:
            * x: tensor to transform.

        Outputs:
            * Transformed tensor.
        """

        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            assert (
                self.min is not None and self.max is not None
            ), "Scaler is not fitted!"
            x_ = (x_ - self.min) / (self.max - self.min + 1e-12)

        elif self.norm == "std":
            assert (
                self.mean is not None and self.std is not None
            ), "Scaler is not fitted!"
            x_ = (x_ - self.mean) / (self.std + 1e-12)

        else:
            raise NotImplementedError

        return x_.reshape(x.transpose(-1, self.dim.item()).shape).transpose(
            -1, self.dim.item()
        )

    @torch.no_grad()
    def fit_transform(self, x):
        """
        Single-step fit scaler and then transform input data.

        Inputs:
            * x: tensor to fit to and transform.

        Outputs:
            * Transformed tensor.
        """
        self.fit(x)
        return self.transform(x)

    @torch.no_grad()
    def inverse_transform(self, x):
        """
        Revert transformation of tensor to return to original coordinates.

        Inputs:
            * x: tensor to de-transform.

        Outputs:
            * De-transformed tensor.
        """

        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            assert (
                self.min is not None and self.max is not None
            ), "Scaler is not fitted!"
            x_ = x_ * (self.max - self.min + 1e-12) + self.min

        elif self.norm == "std":
            assert (
                self.mean is not None and self.std is not None
            ), "Scaler is not fitted!"
            x_ = x_ * (self.std + 1e-12) + self.mean

        else:
            raise NotImplementedError

        return x_.reshape(x.transpose(-1, self.dim.item()).shape).transpose(
            -1, self.dim.item()
        )

    @torch.no_grad()
    def forward(self, x):
        """
        Forward method for scaler (implicitly called via `__call__`).
            Calls `transform` internally.

        Inputs:
            * x: tensor to transform.

        Outputs:
            * Transformed tensor.

        >>> scaler = Scaler(norm='std', dim=-1)
        >>> x = scaler(x0)  # transform tensor `x0`
        """
        return self.transform(x)

    def save(self, filename):
        """
        Save scaler's state (`state_dict`).

        Inputs:
            * filename: path to state file.
        """
        torch.save(deepcopy(self.state_dict()), filename)

    def load(self, filename):
        """
        Load state from file (`state_dict`). Automatically handles initialization parameters (see example).

        Inputs:
            * filename: path to state file.

        >>> f = 'path/to/state/file.pt'
        >>> sc0 = Scaler(norm='minmax', dim=-2)
        >>> sc0.save(f)
        >>> sc1 = Scaler(norm='std', dim=-1)
        >>> sc1.load(f)  # Correctly loads scaler with min-max normalization along dimension -1
        """
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


if __name__ == "__main__":

    rootdir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

    with open(os.path.join(rootdir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    srcdir = os.path.abspath(os.path.join(rootdir, config["paths"]["srcdir"]))
    datadir = os.path.abspath(os.path.join(rootdir, config["paths"]["datadir"]))
    splits = config["data"]["splits"]
    assert sum(splits.values()) <= 1.0, "Invalid split ratios: sum exceeds 1.0"

    if os.path.exists(datadir):
        shutil.rmtree(datadir)
        remove_existing = True
    else:
        remove_existing = False
    os.makedirs(datadir)
    for split_name in ["train", "valid", "test"]:
        os.makedirs(os.path.join(datadir, split_name))

    logging.basicConfig(
        filename=os.path.join(datadir, "datainfo.log"), level=logging.INFO
    )
    if remove_existing:
        logger.warning(f"Removing existing directory: {datadir}")

    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target
    df.to_csv(os.path.join(srcdir, "california_housing.csv"), index=False)

    X = torch.tensor(housing.data, dtype=torch.float32)  
    y = torch.tensor(housing.target, dtype=torch.float32) 

    N = X.shape[0]
    rng = np.random.RandomState(42)
    r = rng.rand(N)

    thr_train = splits["train"]
    thr_valid = splits["train"] + splits["valid"]

    mask_train = r <= thr_train
    mask_valid = (r > thr_train) & (r <= thr_valid)
    mask_test  = (r > thr_valid) & (r <= thr_valid + splits["test"])

    X_train, y_train = X[mask_train], y[mask_train]
    X_valid, y_valid = X[mask_valid], y[mask_valid]
    X_test,  y_test  = X[mask_test],  y[mask_test]

    scaler_cfg = config.get("data", {}).get("scaler", {"norm": "std", "dim": -1})
    scaler = Scaler(norm=scaler_cfg.get("norm", "std"), dim=scaler_cfg.get("dim", -1))

    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test  = scaler.transform(X_test)
    scaler.save(os.path.join(datadir, "scaler.pt"))

    batch_size = config["data"]["batch_size"] 
    base_name = "california_housing"

    def save_split(split_name, Xs, ys):
        count_total = Xs.shape[0]
        if batch_size is None:
            out_path = os.path.join(datadir, split_name, f"{base_name}.npz")
            np.savez(file=out_path,
                     features=Xs.numpy().astype(np.float32),
                     target=ys_.numpy().astype(np.float32).reshape(-1, 1))
            saved = count_total
            discarded = 0
        else:
            num_batches = count_total // batch_size
            usable = num_batches * batch_size
            if num_batches == 0:
                saved = 0
                discarded = count_total
            else:
                Xb = Xs[:usable]
                yb = ys[:usable]
                for b in range(num_batches):
                    xs = Xb[b*batch_size:(b+1)*batch_size]
                    ys_ = yb[b*batch_size:(b+1)*batch_size]
                    out_path = os.path.join(datadir, split_name, f"{base_name}__b{b+1:03d}.npz")
                    np.savez(file=out_path,
                             features=xs.numpy().astype(np.float32),
                             target=ys_.numpy().astype(np.float32).reshape(-1, 1))
                saved = usable
                discarded = count_total - usable

        logger.info(f"{split_name.upper()}: {count_total} samples initially; {saved} saved to disk; discarded={discarded}")
        print(f"{split_name.upper()}: {count_total} samples initially; {saved} saved; discarded={discarded}")
        if discarded > 0 and batch_size is not None:
            logger.warning(f"{split_name.upper()}: {discarded} samples discarded (didn't fit into batches of size {batch_size}).")
            print(f"Warning [{split_name.upper()}]: {discarded} samples discarded (batch_size={batch_size}).")
        return saved

    saved_train = save_split("train", X_train, y_train)
    saved_valid = save_split("valid", X_valid, y_valid)
    saved_test  = save_split("test",  X_test,  y_test)

    logger.info("Final summary (samples saved per split): "
            f"TRAIN={saved_train}, VALID={saved_valid}, TEST={saved_test}")
    print("Final summary ->",
          f"TRAIN={saved_train}, VALID={saved_valid}, TEST={saved_test}")

