import os
import time
import logging
import shutil
import datetime
import yaml
from typing import Any, Sequence

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import Reader
from src.auxiliar import EarlyStopper, longtiming, save_loss_curves
from src.PMNN import PMNNBlock

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
    Extract a scalar from one-element lists.
    """
    if value is None:
        value = default

    if isinstance(value, (list, tuple)) and len(value) == 1:
        if not isinstance(value[0], (list, tuple)):
            return value[0]

    return value


def as_bool(value):
    """
    Convert YAML-compatible values to bool.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}

    return bool(value)


def tag_value(value):
    """
    Create safe strings for folder/file names.
    """
    return str(value).replace(".", "p")


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
    Evaluate one epoch for regression.

    For Diabetes:

        outputs.shape = (batch_size, 1)
        targets.shape = (batch_size, 1)

    The loss used is MSELoss.
    """
    total_loss = 0.0
    total_abs_error = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device=device, dtype=dtype).view(-1, 1)

        outputs = model(inputs)

        if outputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        loss = criterion(outputs, targets)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(outputs - targets).sum().item()
        total_samples += batch_size

    mse = total_loss / max(total_samples, 1)
    rmse = mse ** 0.5
    mae = total_abs_error / max(total_samples, 1)

    return mse, rmse, mae


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
    t_step=None,
    t_end=None,
    batch_size=None,
    lr=None,
):
    """
    Train, validate and test a PMNN model for Diabetes regression.

    The early stopping criterion is validation MSE.
    """
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
    valid_rmses = []
    valid_maes = []
    epoch_times = []

    model = model.to(**factory_kwargs)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_samples = 0

        for inputs, targets in train_dataloader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device, dtype=dtype).view(-1, 1)

            outputs = model(inputs)

            if outputs.shape != targets.shape:
                raise ValueError(
                    f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}"
                )

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            current_batch_size = targets.size(0)
            train_loss_sum += loss.item() * current_batch_size
            train_samples += current_batch_size

        train_loss = train_loss_sum / max(train_samples, 1)
        train_losses.append(train_loss)

        model.eval()

        valid_loss, valid_rmse, valid_mae = _evaluate_epoch(
            model,
            valid_dataloader,
            criterion,
            device=device,
            dtype=dtype,
        )

        valid_losses.append(valid_loss)
        valid_rmses.append(valid_rmse)
        valid_maes.append(valid_mae)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(
            f"TRAIN MSE: {train_loss:.6f} | "
            f"VALID MSE: {valid_loss:.6f} | "
            f"VALID RMSE: {valid_rmse:.6f} | "
            f"VALID MAE: {valid_mae:.6f} | "
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

    best_valid_mse = valid_losses[best_epoch_index]
    best_valid_rmse = valid_rmses[best_epoch_index]
    best_valid_mae = valid_maes[best_epoch_index]

    print("\nTRAINING COMPLETE\nTOTAL TIME:", longtiming(time.time() - start))
    print(f"BEST VALID MSE: {best_valid_mse:.6f}")
    print(f"BEST VALID RMSE: {best_valid_rmse:.6f}")
    print(f"BEST VALID MAE: {best_valid_mae:.6f}")

    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_start = time.time()
    model.eval()

    test_mse, test_rmse, test_mae = _evaluate_epoch(
        model,
        test_dataloader,
        criterion,
        device=device,
        dtype=dtype,
    )

    test_time_s = time.time() - test_start
    test_time_str = longtiming(test_time_s)

    print("\nTEST METRICS:")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"TEST INFERENCE TIME: {test_time_s:.2f}s ({test_time_str})")

    os.makedirs(results_folder, exist_ok=True)

    best_model_path = os.path.join(results_folder, "best_model_state_dict.pth")
    torch.save(model.state_dict(), best_model_path)

    print(f"\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(best_model_path)}\n")
    logger.info(f"Model saved to: {os.path.abspath(best_model_path)}")

    try:
        save_loss_curves(
            train_losses,
            valid_losses,
            t_step,
            t_end,
            batch_size,
            lr,
            results_folder,
        )
    except TypeError:
        save_loss_curves(
            train_losses,
            valid_losses,
            batch_size,
            lr,
            results_folder,
        )

    validation_metrics_path = os.path.join(results_folder, "validation_metrics.txt")

    with open(validation_metrics_path, "w") as f:
        f.write("epoch, train_mse, valid_mse, valid_rmse, valid_mae\n")
        for idx, (tr_loss, va_loss, va_rmse, va_mae) in enumerate(
            zip(train_losses, valid_losses, valid_rmses, valid_maes),
            start=1,
        ):
            f.write(f"{idx}, {tr_loss}, {va_loss}, {va_rmse}, {va_mae}\n")

    elapsed = time.time() - start
    elapsed_str = longtiming(elapsed)

    return (
        best_valid_mse,
        best_valid_rmse,
        best_valid_mae,
        train_losses,
        valid_losses,
        valid_rmses,
        valid_maes,
        test_mse,
        test_rmse,
        test_mae,
        elapsed_str,
        epoch_times,
    )


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))
    baserundir = os.path.join(rootdir, "runs_pmnn_diabetes")
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

    best_val_mse = float("inf")
    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    best_test_mse = float("inf")
    best_test_rmse = float("inf")
    best_test_mae = float("inf")
    best_params = None
    best_config_name = None

    results_folder = rundir
    os.makedirs(results_folder, exist_ok=True)

    hyperparameter_results_filename = os.path.join(
        results_folder,
        "hyperparameter_results.txt",
    )

    with open(hyperparameter_results_filename, "w") as result_file:
        result_file.write(
            "File Name, Best Valid MSE, Best Valid RMSE, Best Valid MAE, "
            "Test MSE, Test RMSE, Test MAE, "
            "t_step, t_end, batch_size, lr, input_size, hidden_size, output_size\n"
        )

    input_size = int(unwrap_scalar(config["model"].get("input_size", 10)))
    hidden_size = int(unwrap_scalar(config["model"].get("hidden_size", 2)))
    output_size = int(unwrap_scalar(config["model"].get("output_size", 1)))

    t_step_values: Sequence[int] = as_list(config["model"].get("t_step", [10]))
    t_end_values: Sequence[float] = as_list(config["model"].get("t_end", [10.0]))

    batch_size_values: Sequence[int] = as_list(
        config["model"].get(
            "batch_size",
            config.get("loaders", {}).get("batch_size", 64),
        )
    )

    lr_values: Sequence[float] = as_list(config["optimizer"]["lr"])

    optimizer_name = str(config["optimizer"].get("name", "Adam")).lower()
    weight_decay = float(config["optimizer"].get("weight_decay", 0.0))
    momentum = float(config["optimizer"].get("momentum", 0.9))
    betas = tuple(config["optimizer"].get("betas", [0.9, 0.999]))

    a = float(unwrap_scalar(config["model"].get("a", 0.2)))
    b = float(unwrap_scalar(config["model"].get("b", 0.02)))
    g = float(unwrap_scalar(config["model"].get("g", 3.0)))
    I = float(unwrap_scalar(config["model"].get("I", 0.0)))

    initial_state_scale = float(
        unwrap_scalar(config["model"].get("initial_state_scale", 1.0))
    )

    state_clip = config["model"].get("state_clip", 20.0)

    if state_clip is not None:
        state_clip = float(unwrap_scalar(state_clip))

    use_layer_norm = as_bool(config["model"].get("use_layer_norm", True))

    for t_step in t_step_values:
        for t_end in t_end_values:
            for batch_size in batch_size_values:
                for lr in lr_values:
                    t_step = int(t_step)
                    t_end = float(t_end)
                    batch_size = int(batch_size)
                    lr = float(lr)

                    print(
                        f"Evaluating t_step = {t_step}, "
                        f"t_end = {t_end}, "
                        f"batch_size = {batch_size}, "
                        f"lr = {lr}, "
                        f"input_size = {input_size}, "
                        f"hidden_size = {hidden_size}, "
                        f"output_size = {output_size}"
                    )

                    model = PMNNBlock(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        t_step=t_step,
                        t_end=t_end,
                        a=a,
                        b=b,
                        g=g,
                        I=I,
                        initial_state_scale=initial_state_scale,
                        state_clip=state_clip,
                        use_layer_norm=use_layer_norm,
                        batch_size=batch_size,
                        device=DEVICE,
                        dtype=DTYPE,
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
                        f"pmnn_tstep_{tag_value(t_step)}"
                        f"_tend_{tag_value(t_end)}"
                        f"_batch_size_{tag_value(batch_size)}"
                        f"_lr_{tag_value(lr)}"
                    )

                    config_results_folder = os.path.join(rundir, config_name)
                    os.makedirs(config_results_folder, exist_ok=True)

                    (
                        val_mse,
                        val_rmse,
                        val_mae,
                        train_losses,
                        val_losses,
                        val_rmses,
                        val_maes,
                        test_mse,
                        test_rmse,
                        test_mae,
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
                            f"{config_name}__CKPT.pt",
                        ),
                        device=DEVICE,
                        dtype=DTYPE,
                        print_model_summary=True,
                        t_step=t_step,
                        t_end=t_end,
                        batch_size=batch_size,
                        lr=lr,
                    )

                    print(f"Validation MSE: {val_mse}")
                    print(f"Validation RMSE: {val_rmse}")
                    print(f"Validation MAE: {val_mae}")

                    log_filename = os.path.join(
                        config_results_folder,
                        f"{config_name}.txt",
                    )

                    with open(log_filename, "w") as log_file:
                        log_file.write(
                            f"Evaluating parameters: "
                            f"t_step = {t_step}, "
                            f"t_end = {t_end}, "
                            f"batch_size = {batch_size}, "
                            f"lr = {lr}, "
                            f"input_size = {input_size}, "
                            f"hidden_size = {hidden_size}, "
                            f"output_size = {output_size}, "
                            f"a = {a}, "
                            f"b = {b}, "
                            f"g = {g}, "
                            f"I = {I}, "
                            f"initial_state_scale = {initial_state_scale}, "
                            f"state_clip = {state_clip}, "
                            f"use_layer_norm = {use_layer_norm}\n"
                        )
                        log_file.write(f"Best Validation MSE: {val_mse}\n")
                        log_file.write(f"Best Validation RMSE: {val_rmse}\n")
                        log_file.write(f"Best Validation MAE: {val_mae}\n")
                        log_file.write(f"Final Test MSE: {test_mse}\n")
                        log_file.write(f"Final Test RMSE: {test_rmse}\n")
                        log_file.write(f"Final Test MAE: {test_mae}\n")
                        log_file.write(f"Training MSE per epoch: {train_losses}\n")
                        log_file.write(f"Validation MSE per epoch: {val_losses}\n")
                        log_file.write(f"Validation RMSE per epoch: {val_rmses}\n")
                        log_file.write(f"Validation MAE per epoch: {val_maes}\n")
                        log_file.write(f"Training Time (h:m:s): {elapsed_str}\n")
                        log_file.write(f"Epoch Times (seconds): {epoch_times}\n")

                    with open(hyperparameter_results_filename, "a") as result_file:
                        result_file.write(
                            f"{config_name}.txt, "
                            f"{val_mse}, {val_rmse}, {val_mae}, "
                            f"{test_mse}, {test_rmse}, {test_mae}, "
                            f"{t_step}, {t_end}, {batch_size}, {lr}, "
                            f"{input_size}, {hidden_size}, {output_size}\n"
                        )

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_val_rmse = val_rmse
                        best_val_mae = val_mae
                        best_test_mse = test_mse
                        best_test_rmse = test_rmse
                        best_test_mae = test_mae
                        best_params = (
                            t_step,
                            t_end,
                            batch_size,
                            lr,
                            input_size,
                            hidden_size,
                            output_size,
                        )
                        best_config_name = config_name

    if best_params is not None:
        (
            best_t_step,
            best_t_end,
            best_batch_size,
            best_lr,
            best_input_size,
            best_hidden_size,
            best_output_size,
        ) = best_params

        print(f"\nBest parameters found based on validation MSE:")
        print(f"t_step: {best_t_step}")
        print(f"t_end: {best_t_end}")
        print(f"batch_size: {best_batch_size}")
        print(f"learning rate: {best_lr}")
        print(f"input_size: {best_input_size}")
        print(f"hidden_size: {best_hidden_size}")
        print(f"output_size: {best_output_size}")
        print(f"Best validation MSE: {best_val_mse}")
        print(f"Best validation RMSE: {best_val_rmse}")
        print(f"Best validation MAE: {best_val_mae}")
        print(f"Test MSE for this config: {best_test_mse}")
        print(f"Test RMSE for this config: {best_test_rmse}")
        print(f"Test MAE for this config: {best_test_mae}")

        src_best_folder = os.path.join(rundir, best_config_name)
        dst_best_folder = os.path.join(rundir, "best_hparams")

        if os.path.exists(dst_best_folder):
            shutil.rmtree(dst_best_folder)

        shutil.copytree(src_best_folder, dst_best_folder)

        with open(os.path.join(dst_best_folder, "best_summary.txt"), "w") as fsum:
            fsum.write("Best hyperparameters found based on validation MSE:\n")
            fsum.write(f"t_step: {best_t_step}\n")
            fsum.write(f"t_end: {best_t_end}\n")
            fsum.write(f"batch_size: {best_batch_size}\n")
            fsum.write(f"learning rate: {best_lr}\n")
            fsum.write(f"input_size: {best_input_size}\n")
            fsum.write(f"hidden_size: {best_hidden_size}\n")
            fsum.write(f"output_size: {best_output_size}\n")
            fsum.write(f"best validation MSE: {best_val_mse}\n")
            fsum.write(f"best validation RMSE: {best_val_rmse}\n")
            fsum.write(f"best validation MAE: {best_val_mae}\n")
            fsum.write(f"test MSE (best config): {best_test_mse}\n")
            fsum.write(f"test RMSE (best config): {best_test_rmse}\n")
            fsum.write(f"test MAE (best config): {best_test_mae}\n")

    else:
        print("No valid configuration found during the sweep.")

