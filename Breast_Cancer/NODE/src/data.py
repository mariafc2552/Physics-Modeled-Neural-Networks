import shutil
import torch
import logging
import yaml
import numpy as np
import os
from copy import deepcopy
from glob import glob
from typing import List
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

logger = logging.getLogger(__name__)

np.random.seed(42)


class Reader:
    """
    Class for reading MNIST dataset splits from directory.

    The reader expects preprocessed MNIST batches stored as .npz files. Each file
    contains a batch of flattened grayscale images and their corresponding integer
    class labels.

    Expected input representation:
        * features: tensor of shape (batch_size, 784)
        * target: tensor of shape (batch_size,)

    The target values are integer class indices in [0, 9], which makes the dataset
    directly compatible with `torch.nn.CrossEntropyLoss`.
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
                │   ├── mnist__b001.npz
                │   ├── mnist__b002.npz
                │   └── ...
                ├── valid/
                │   └── ...
                └── test/
                    └── ...

            Each .npz contains:
                'features': array of shape (batch_size, 784)
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
            * f: path to .npz file.

        Outputs:
            * features: torch.FloatTensor with shape (batch_size, 784).
            * target: torch.LongTensor with shape (batch_size,).
        """
        data = dict(np.load(f))

        features = torch.tensor(data["features"], dtype=torch.float32)
        target = torch.tensor(data["target"], dtype=torch.long)

        return features, target

    def __call__(self, key=None):
        """
        Get file list for given split key.

        Inputs:
            * key: 'train', 'valid' or 'test'. If None, returns dict of all splits.

        Outputs:
            * list of files if key is given, or dict of lists if key is None.
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
            * key: 'train', 'valid' or 'test'.

        Outputs:
            * TensorDataset containing concatenated features and targets.
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
            * batch_size: size of the mini-batches.

        Outputs:
            * train_loader, val_loader, test_loader.
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
    Scaler class to implement standardization or min-max normalization.

    This class is kept for compatibility with previous experiments. For MNIST, the
    usual preprocessing is to scale pixel intensities from [0, 255] to [0, 1].
    Therefore, the scaler is not strictly required unless additional feature-level
    standardization is desired.
    """

    def __init__(self, norm="std", *, dim=-1):
        """
        Initializer for `Scaler`.

        Inputs:
            * norm: 'std' for standardization or 'minmax' for min-max normalization.
            * dim: feature dimension.
        """
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
        """
        Reset scaler to defaults.
        """
        self.register_buffer("min", None)
        self.register_buffer("max", None)
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.cumlen = torch.tensor(0, dtype=torch.int64)

    @torch.no_grad()
    def fit(self, x):
        """
        Fit scaler in a single step.

        Inputs:
            * x: tensor to fit.
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
            * transformed tensor.
        """
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
            -1, self.dim.item()
        )

    @torch.no_grad()
    def fit_transform(self, x):
        """
        Fit scaler and transform tensor.

        Inputs:
            * x: tensor to fit and transform.

        Outputs:
            * transformed tensor.
        """
        self.fit(x)
        return self.transform(x)

    @torch.no_grad()
    def inverse_transform(self, x):
        """
        Revert scaler transformation.

        Inputs:
            * x: tensor to inverse-transform.

        Outputs:
            * tensor in original coordinates.
        """
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
            -1, self.dim.item()
        )

    @torch.no_grad()
    def forward(self, x):
        """
        Forward method for scaler.
        """
        return self.transform(x)

    def save(self, filename):
        """
        Save scaler state.
        """
        torch.save(deepcopy(self.state_dict()), filename)

    def load(self, filename):
        """
        Load scaler state.
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

    os.makedirs(srcdir, exist_ok=True)

    splits = config["data"]["splits"]

    if os.path.exists(datadir):
        shutil.rmtree(datadir)
        remove_existing = True
    else:
        remove_existing = False

    os.makedirs(datadir)

    for split_name in ["train", "valid", "test"]:
        os.makedirs(os.path.join(datadir, split_name))

    logging.basicConfig(
        filename=os.path.join(datadir, "datainfo.log"),
        level=logging.INFO,
    )

    if remove_existing:
        logger.warning(f"Removing existing directory: {datadir}")

    # -------------------------------------------------------------------------
    # Load MNIST
    # -------------------------------------------------------------------------
    # Official MNIST split:
    #   * train=True  -> 60,000 samples
    #   * train=False -> 10,000 samples
    #
    # We split the official training set into train and validation subsets.
    # The official test set is preserved as test.
    # -------------------------------------------------------------------------

    mnist_train = MNIST(
        root=srcdir,
        train=True,
        download=True,
    )

    mnist_test = MNIST(
        root=srcdir,
        train=False,
        download=True,
    )

    X_full_train = mnist_train.data.float() / 255.0
    y_full_train = mnist_train.targets.long()

    X_test = mnist_test.data.float() / 255.0
    y_test = mnist_test.targets.long()

    # Flatten images:
    #   original shape: (N, 28, 28)
    #   flattened shape: (N, 784)
    X_full_train = X_full_train.reshape(X_full_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # -------------------------------------------------------------------------
    # Train / validation split from the official MNIST training partition
    # -------------------------------------------------------------------------

    N_train_full = X_full_train.shape[0]

    rng = np.random.RandomState(42)
    perm = rng.permutation(N_train_full)

    train_ratio = float(splits.get("train", 0.8))
    valid_ratio = float(splits.get("valid", 0.2))

    ratio_sum = train_ratio + valid_ratio

    train_ratio = train_ratio / ratio_sum
    valid_ratio = valid_ratio / ratio_sum

    n_train = int(N_train_full * train_ratio)

    idx_train = perm[:n_train]
    idx_valid = perm[n_train:]

    X_train = X_full_train[idx_train]
    y_train = y_full_train[idx_train]

    X_valid = X_full_train[idx_valid]
    y_valid = y_full_train[idx_valid]

    # -------------------------------------------------------------------------
    # Optional additional standardization
    # -------------------------------------------------------------------------
    # MNIST pixels are already scaled to [0, 1]. For a strict image-classification
    # pipeline, this is enough. If config["data"]["scaler"]["use"] is true, then
    # an additional feature-wise scaler is fitted on the training set only.
    # -------------------------------------------------------------------------

    scaler_cfg = config.get("data", {}).get("scaler", None)
    use_scaler = False

    if isinstance(scaler_cfg, dict):
        use_scaler = bool(scaler_cfg.get("use", False))

    if use_scaler:
        scaler = Scaler(
            norm=scaler_cfg.get("norm", "std"),
            dim=scaler_cfg.get("dim", -1),
        )

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        scaler.save(os.path.join(datadir, "scaler.pt"))

    # -------------------------------------------------------------------------
    # Save split batches
    # -------------------------------------------------------------------------

    batch_size = config["data"]["batch_size"]
    base_name = "mnist"

    def save_split(split_name, Xs, ys):
        """
        Save one split as a sequence of .npz mini-batches.

        Inputs:
            * split_name: train, valid or test.
            * Xs: features.
            * ys: labels.
        """
        count_total = Xs.shape[0]

        if batch_size is None:
            out_path = os.path.join(datadir, split_name, f"{base_name}.npz")

            np.savez(
                file=out_path,
                features=Xs.numpy().astype(np.float32),
                target=ys.numpy().astype(np.int64),
            )

            saved = count_total
            discarded = 0

        else:
            bsz = int(batch_size)
            num_batches = count_total // bsz
            usable = num_batches * bsz

            if num_batches == 0:
                saved = 0
                discarded = count_total

            else:
                Xb = Xs[:usable]
                yb = ys[:usable]

                for b in range(num_batches):
                    xs = Xb[b * bsz:(b + 1) * bsz]
                    ys_ = yb[b * bsz:(b + 1) * bsz]

                    out_path = os.path.join(
                        datadir,
                        split_name,
                        f"{base_name}__b{b + 1:03d}.npz",
                    )

                    np.savez(
                        file=out_path,
                        features=xs.numpy().astype(np.float32),
                        target=ys_.numpy().astype(np.int64),
                    )

                saved = usable
                discarded = count_total - usable

        logger.info(
            f"{split_name.upper()}: {count_total} samples initially; "
            f"{saved} saved to disk; discarded={discarded}"
        )

        print(
            f"{split_name.upper()}: {count_total} samples initially; "
            f"{saved} saved; discarded={discarded}"
        )

        if discarded > 0 and batch_size is not None:
            logger.warning(
                f"{split_name.upper()}: {discarded} samples discarded "
                f"because they did not fit into batches of size {batch_size}."
            )

            print(
                f"Warning [{split_name.upper()}]: {discarded} samples discarded "
                f"(batch_size={batch_size})."
            )

        return saved

    saved_train = save_split("train", X_train, y_train)
    saved_valid = save_split("valid", X_valid, y_valid)
    saved_test = save_split("test", X_test, y_test)

    logger.info(
        "Final summary: "
        f"TRAIN={saved_train}, VALID={saved_valid}, TEST={saved_test}"
    )

    print(
        "Final summary ->",
        f"TRAIN={saved_train}, VALID={saved_valid}, TEST={saved_test}",
    )