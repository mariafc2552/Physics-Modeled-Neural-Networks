import os
import time
import logging
import shutil
import datetime
import yaml
from typing import Sequence, Any

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import Reader
from src.auxiliar import EarlyStopper, longtiming, save_loss_curves
from src.CFC import CFCblock

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def as_list(value):
    """
    Convert scalar values to one-element lists for grid-search compatibility.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def unwrap_scalar(value: Any, default=None):
    """
    Extract a scalar from one-element grid-search lists.
    """
    if value is None:
        value = default

    if isinstance(value, (list, tuple)) and len(value) == 1:
        if not isinstance(value[0], (list, tuple)):
            return value[0]

    return value


@torch.no_grad()
def _evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype = DTYPE,
):
    """
    Evaluate one epoch for Energy Efficiency regression.

    The CfC model returns continuous predictions:

        outputs.shape = (batch_size, 2)

    The two target variables are:

        targets[:, 0] -> heating_load
        targets[:, 1] -> cooling_load

    Therefore, the loss is MSELoss.
    """
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device=device, dtype=dtype)

        outputs = model(inputs)

        if targets.dim() == 1:
            targets = targets.view(-1, outputs.size(1))

        if outputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        loss = criterion(outputs, targets)

        current_batch_size = inputs.size(0)
        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

    return total_loss / max(total_samples, 1)


def fit(
    model,
    *,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    optimizer,
    num_epochs,
    patience,
    min_delta,
    results_folder,
    checkpoint_path=None,
    device=DEVICE,
    dtype=DTYPE,
    print_model_summary=True,
    num_units=None,
    batch_size=None,
    lr=None,
):

    factory_kwargs = {"device": device, "dtype": dtype}

    start = time.time()
    now = datetime.datetime.now()
    dt = now.strftime("%d-%m-%Y %H:%M:%S")
    print(f"\nSTARTING @ {dt}\n")

    if print_model_summary:
        total_params = sum(p.numel() for p in model.parameters())
        print("\nMODEL DESCRIPTION:\n\n", model, "\n\n")
        print(f"Total number of model parameters: {total_params}\n")

    criterion = nn.MSELoss()
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    train_losses = []
    valid_losses = []
    valid_rmse_values = []
    epoch_times = []

    model = model.to(**factory_kwargs)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        train_samples = 0

        for inputs, targets in train_dataloader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device, dtype=dtype)

            outputs = model(inputs)

            if targets.dim() == 1:
                targets = targets.view(-1, outputs.size(1))

            if outputs.shape != targets.shape:
                raise ValueError(
                    f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}"
                )

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            current_batch_size = inputs.size(0)
            train_loss += loss.item() * current_batch_size
            train_samples += current_batch_size

        train_loss /= max(train_samples, 1)
        train_losses.append(train_loss)

        model.eval()

        valid_loss = _evaluate_epoch(
            model,
            valid_dataloader,
            criterion,
            device=device,
            dtype=dtype,
        )

        valid_rmse = valid_loss ** 0.5

        valid_losses.append(valid_loss)
        valid_rmse_values.append(valid_rmse)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(
            f"TRAIN LOSS (MSE): {train_loss:.6f} | "
            f"VALID LOSS (MSE): {valid_loss:.6f} | "
            f"VALID RMSE: {valid_rmse:.6f} | "
            f"EPOCH TIME: {epoch_time:.2f}s"
        )

        stop = stopper(valid_loss, model.state_dict())

        if checkpoint_path is not None:
            stopper.save(checkpoint_path)

        if stop:
            print("\nEARLY STOPPING\n")
            break

    if stopper.best_model is not None:
        model.load_state_dict(stopper.best_model)

    best_epoch_index = min(
        range(len(valid_losses)),
        key=lambda idx: valid_losses[idx],
    )

    best_valid_loss = valid_losses[best_epoch_index]
    best_valid_rmse = valid_rmse_values[best_epoch_index]

    print("\nTRAINING COMPLETE\nTOTAL TIME:", longtiming(time.time() - start))
    print(f"BEST VALID LOSS (MSE): {best_valid_loss:.6f}")
    print(f"BEST VALID RMSE: {best_valid_rmse:.6f}")

    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_start = time.time()
    model.eval()

    test_loss = _evaluate_epoch(
        model,
        test_dataloader,
        criterion,
        device=device,
        dtype=dtype,
    )

    test_rmse = test_loss ** 0.5

    test_time_s = time.time() - test_start
    test_time_str = longtiming(test_time_s)

    print("\nTEST METRICS:")
    print(f"Test MSE Loss: {test_loss:.6f}")
    print(f"Test RMSE Loss: {test_rmse:.6f}")
    print(f"TEST INFERENCE TIME: {test_time_s:.2f}s ({test_time_str})")

    os.makedirs(results_folder, exist_ok=True)

    best_model_path = os.path.join(results_folder, "best_model_state_dict.pth")
    torch.save(model.state_dict(), best_model_path)

    print(f"\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(best_model_path)}\n")
    logger.info(f"Model saved to: {os.path.abspath(best_model_path)}")

    save_loss_curves(
        train_losses,
        valid_losses,
        num_units,
        batch_size,
        lr,
        results_folder,
    )

    validation_metrics_path = os.path.join(results_folder, "validation_metrics.txt")

    with open(validation_metrics_path, "w") as f:
        f.write("epoch, train_loss_mse, valid_loss_mse, valid_rmse\n")
        for idx, (tr_loss, va_loss, va_rmse) in enumerate(
            zip(train_losses, valid_losses, valid_rmse_values),
            start=1,
        ):
            f.write(f"{idx}, {tr_loss}, {va_loss}, {va_rmse}\n")

    elapsed = time.time() - start
    elapsed_str = longtiming(elapsed)

    return (
        best_valid_loss,
        best_valid_rmse,
        train_losses,
        valid_losses,
        valid_rmse_values,
        test_loss,
        test_rmse,
        elapsed_str,
        epoch_times,
    )


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))
    baserundir = os.path.join(rootdir, "runs_cfc_energy_efficiency")
    os.makedirs(baserundir, exist_ok=True)

    with open(os.path.join(rootdir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

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

    print(f"DEVICE: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA no disponible. Se usará CPU.")

    datadir = config["paths"]["datadir"]
    reader = Reader(datadir)

    train_dataset = reader.make_dataset("train")
    val_dataset = reader.make_dataset("valid")
    test_dataset = reader.make_dataset("test")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_test_loss = float("inf")
    best_test_rmse = float("inf")
    best_params = None

    results_folder = rundir
    os.makedirs(results_folder, exist_ok=True)

    hyperparameter_results_filename = os.path.join(
        results_folder,
        "hyperparameter_results.txt",
    )

    with open(hyperparameter_results_filename, "w") as result_file:
        result_file.write(
            "File Name, Best Valid MSE, Best Valid RMSE, "
            "Test MSE, Test RMSE, "
            "Num Units, Batch Size, LR, Input Size, Output Size\n"
        )

    input_size = int(unwrap_scalar(config["model"].get("input_size", 8)))
    output_size = int(unwrap_scalar(config["model"].get("output_size", 2)))

    if input_size != 8:
        print(
            f"Warning: Energy Efficiency usually has input_size=8, "
            f"but input_size={input_size} was provided."
        )

    if output_size != 2:
        print(
            f"Warning: Energy Efficiency with Heating Load and Cooling Load "
            f"usually has output_size=2, but output_size={output_size} was provided."
        )

    batch_size_values: Sequence[int] = as_list(
        config["model"].get(
            "batch_size",
            config.get("loaders", {}).get("batch_size", 32),
        )
    )

    lr_values: Sequence[float] = as_list(config["optimizer"]["lr"])

    num_units_values: Sequence[int] = as_list(
        config["model"].get(
            "num_units_values",
            config["model"].get("num_units", [64]),
        )
    )

    optimizer_name = str(config["optimizer"].get("name", "Adam")).lower()
    weight_decay = float(config["optimizer"].get("weight_decay", 0.0))
    momentum = float(config["optimizer"].get("momentum", 0.9))
    betas = tuple(config["optimizer"].get("betas", [0.9, 0.999]))

    for num_units in num_units_values:
        for batch_size in batch_size_values:
            for lr in lr_values:
                num_units = int(num_units)
                batch_size = int(batch_size)
                lr = float(lr)

                print(
                    f"Evaluating num_units = {num_units}, "
                    f"batch_size = {batch_size}, "
                    f"lr = {lr}, "
                    f"input_size = {input_size}, "
                    f"output_size = {output_size}"
                )

                model = CFCblock(
                    input_size=input_size,
                    num_units=num_units,
                    output_size=output_size,
                )

                model = model.to(device=DEVICE, dtype=DTYPE)

                if optimizer_name == "sgd":
                    optimizer = optim.SGD(
                        model.parameters(),
                        lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
                    )

                elif optimizer_name == "adam":
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=lr,
                        weight_decay=weight_decay,
                        betas=betas,
                    )

                else:
                    raise ValueError(
                        "Invalid optimizer name. Expected 'SGD' or 'Adam'."
                    )

                train_loader, val_loader, test_loader = reader.make_loaders(batch_size)

                config_name = (
                    f"num_units_{num_units}"
                    f"_batch_size_{batch_size}"
                    f"_lr_{lr}"
                )

                config_results_folder = os.path.join(rundir, config_name)
                os.makedirs(config_results_folder, exist_ok=True)

                (
                    val_loss,
                    val_rmse,
                    train_losses,
                    val_losses,
                    val_rmse_values,
                    test_loss,
                    test_rmse,
                    elapsed_str,
                    epoch_times,
                ) = fit(
                    model=model,
                    train_dataloader=train_loader,
                    valid_dataloader=val_loader,
                    test_dataloader=test_loader,
                    optimizer=optimizer,
                    num_epochs=int(config["fit"]["num_epochs"]),
                    patience=int(config["fit"]["patience"]),
                    min_delta=float(config["fit"]["min_delta"]),
                    results_folder=config_results_folder,
                    checkpoint_path=os.path.join(
                        config_results_folder,
                        f"cfc_neuron_{num_units}_bs{batch_size}_lr{lr}__CKPT.pt",
                    ),
                    device=DEVICE,
                    dtype=DTYPE,
                    print_model_summary=True,
                    num_units=num_units,
                    batch_size=batch_size,
                    lr=lr,
                )

                print(f"Validation MSE Loss: {val_loss}")
                print(f"Validation RMSE Loss: {val_rmse}")

                log_filename = os.path.join(
                    config_results_folder,
                    f"{config_name}.txt",
                )

                with open(log_filename, "w") as log_file:
                    log_file.write(
                        f"Evaluating parameters: "
                        f"num_units = {num_units}, "
                        f"batch_size = {batch_size}, "
                        f"lr = {lr}, "
                        f"input_size = {input_size}, "
                        f"output_size = {output_size}\n"
                    )
                    log_file.write(f"Best Validation MSE Loss: {val_loss}\n")
                    log_file.write(f"Best Validation RMSE Loss: {val_rmse}\n")
                    log_file.write(f"Final Test MSE Loss: {test_loss}\n")
                    log_file.write(f"Final Test RMSE Loss: {test_rmse}\n")
                    log_file.write(f"Training Loss per epoch: {train_losses}\n")
                    log_file.write(f"Validation Loss per epoch: {val_losses}\n")
                    log_file.write(f"Validation RMSE per epoch: {val_rmse_values}\n")
                    log_file.write(f"Training Time (h:m:s): {elapsed_str}\n")
                    log_file.write(f"Epoch Times (seconds): {epoch_times}\n")

                with open(hyperparameter_results_filename, "a") as result_file:
                    result_file.write(
                        f"{config_name}.txt, "
                        f"{val_loss}, {val_rmse}, "
                        f"{test_loss}, {test_rmse}, "
                        f"{num_units}, {batch_size}, {lr}, "
                        f"{input_size}, {output_size}\n"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_rmse = val_rmse
                    best_test_loss = test_loss
                    best_test_rmse = test_rmse
                    best_params = (
                        num_units,
                        batch_size,
                        lr,
                        input_size,
                        output_size,
                    )

    if best_params is not None:
        (
            best_num_units,
            best_batch_size,
            best_lr,
            best_input_size,
            best_output_size,
        ) = best_params

        print(f"\nBest parameters found:")
        print(f"num_units: {best_num_units}")
        print(f"batch_size: {best_batch_size}")
        print(f"learning rate: {best_lr}")
        print(f"input_size: {best_input_size}")
        print(f"output_size: {best_output_size}")
        print(f"Best validation_loss MSE: {best_val_loss}")
        print(f"Best validation_loss RMSE: {best_val_rmse}")
        print(f"Test MSE loss for this config: {best_test_loss}")
        print(f"Test RMSE loss for this config: {best_test_rmse}")

        best_config_foldername = (
            f"num_units_{best_num_units}"
            f"_batch_size_{best_batch_size}"
            f"_lr_{best_lr}"
        )

        src_best_folder = os.path.join(rundir, best_config_foldername)
        dst_best_folder = os.path.join(rundir, "best_hparams")

        if os.path.exists(dst_best_folder):
            shutil.rmtree(dst_best_folder)

        shutil.copytree(src_best_folder, dst_best_folder)

        with open(os.path.join(dst_best_folder, "best_summary.txt"), "w") as fsum:
            fsum.write("mejor configuración de hiperparámetros\n")
            fsum.write(f"num_units: {best_num_units}\n")
            fsum.write(f"batch_size: {best_batch_size}\n")
            fsum.write(f"learning rate: {best_lr}\n")
            fsum.write(f"input_size: {best_input_size}\n")
            fsum.write(f"output_size: {best_output_size}\n")
            fsum.write(f"best validation_loss MSE: {best_val_loss}\n")
            fsum.write(f"best validation_loss RMSE: {best_val_rmse}\n")
            fsum.write(f"test MSE loss (mejor config): {best_test_loss}\n")
            fsum.write(f"test RMSE loss (mejor config): {best_test_rmse}\n")

    else:
        print("No valid configuration found during the sweep.")

