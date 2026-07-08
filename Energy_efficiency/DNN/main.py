import os
import time
import logging
import shutil
import datetime
import yaml
from typing import Sequence

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import Reader
from src.auxiliar import EarlyStopper, longtiming, save_loss_curves
from src.DNN import DNNBlock

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@torch.no_grad()
def _evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    *,
    device: torch.device,
):
    total = 0.0

    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=DTYPE)
        targets = targets.to(device=device, dtype=DTYPE)

        outputs = model(inputs)

        if targets.dim() == 1:
            targets = targets.view(-1, outputs.size(1))

        if outputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        loss = criterion(outputs, targets)
        total += loss.item()

    return total / max(len(dataloader), 1)


def _normalize_hidden_sizes_grid(hidden_sizes_config):
    """
    Normalize the hidden_sizes block from the YAML configuration.

    Accepted formats:
        hidden_sizes: [64, 32, 16]
        hidden_sizes: [[64, 32, 16], [128, 64, 32], [128, 128]]

    Returns:
        List of tuples, where each tuple is one DNN architecture.
    """
    if hidden_sizes_config is None:
        raise ValueError("model.hidden_sizes must be defined in the configuration file.")

    if len(hidden_sizes_config) == 0:
        raise ValueError("model.hidden_sizes must not be empty.")

    first_element = hidden_sizes_config[0]

    if isinstance(first_element, int):
        return [tuple(int(h) for h in hidden_sizes_config)]

    hidden_sizes_grid = []

    for hidden_sizes in hidden_sizes_config:
        if hidden_sizes is None or len(hidden_sizes) == 0:
            raise ValueError(
                f"Invalid hidden_sizes architecture found: {hidden_sizes}"
            )

        hidden_sizes_tuple = tuple(int(h) for h in hidden_sizes)

        if any(h <= 0 for h in hidden_sizes_tuple):
            raise ValueError(
                f"All hidden layer sizes must be positive. Received: {hidden_sizes_tuple}"
            )

        hidden_sizes_grid.append(hidden_sizes_tuple)

    return hidden_sizes_grid


def _as_sequence(value):
    """
    Convert a scalar or a list from the YAML file into a Python list.

    This allows using either:
        lr: [0.001, 0.0005]
    or:
        weight_decay: 0.005
    """
    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def get_activation(activation_name):
    """
    Return activation class from a string name.
    """
    activation_name = str(activation_name).lower()

    if activation_name == "relu":
        return nn.ReLU

    if activation_name == "silu":
        return nn.SiLU

    if activation_name == "tanh":
        return nn.Tanh

    if activation_name == "gelu":
        return nn.GELU

    raise ValueError(
        f"Unknown activation '{activation_name}'. "
        "Valid options are: relu, silu, tanh, gelu."
    )


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
    batch_size=None,
    lr=None,
    hidden_sizes=None,
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
    epoch_times = []

    model = model.to(**factory_kwargs)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

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

            train_loss += loss.item()

        train_loss /= max(len(train_dataloader), 1)
        train_losses.append(train_loss)

        model.eval()
        valid_loss = _evaluate_epoch(
            model,
            valid_dataloader,
            criterion,
            device=device,
        )
        valid_losses.append(valid_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(
            f"TRAIN LOSS (MSE): {train_loss:.6f} | "
            f"VALID LOSS (MSE): {valid_loss:.6f} | "
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

    print("\nTRAINING COMPLETE\nTOTAL TIME:", longtiming(time.time() - start))

    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_start = time.time()
    model.eval()

    test_mse = _evaluate_epoch(
        model,
        test_dataloader,
        criterion,
        device=device,
    )
    test_rmse = test_mse ** 0.5

    test_time_s = time.time() - test_start
    test_time_str = longtiming(test_time_s)

    print("\nTEST METRICS:")
    print(f"RMSE Test Loss: {test_rmse:.6f}")
    print(f"TEST INFERENCE TIME: {test_time_s:.2f}s ({test_time_str})")

    os.makedirs(results_folder, exist_ok=True)

    best_model_path = os.path.join(results_folder, "best_model_state_dict.pth")
    torch.save(model.state_dict(), best_model_path)

    print(f"\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(best_model_path)}\n")
    logger.info(f"Model saved to: {os.path.abspath(best_model_path)}")

    save_loss_curves(
        train_losses,
        valid_losses,
        batch_size,
        lr,
        results_folder,
    )

    elapsed = time.time() - start
    elapsed_str = longtiming(elapsed)

    best_valid_loss = min(valid_losses) if len(valid_losses) > 0 else float("inf")

    return (
        best_valid_loss,
        train_losses,
        valid_losses,
        test_rmse,
        elapsed_str,
        epoch_times,
    )


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))
    baserundir = os.path.join(rootdir, "runs_dnn_energy_efficiency")
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

    datadir = config["paths"]["datadir"]

    reader = Reader(datadir)

    train_dataset = reader.make_dataset("train")
    val_dataset = reader.make_dataset("valid")
    test_dataset = reader.make_dataset("test")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    best_val_loss = float("inf")
    best_test_loss = float("inf")
    best_params = None

    results_folder = rundir
    os.makedirs(results_folder, exist_ok=True)

    hyperparameter_results_filename = os.path.join(
        results_folder,
        "hyperparameter_results.txt",
    )

    with open(hyperparameter_results_filename, "w") as result_file:
        result_file.write(
            "hidden_sizes,batch_size,learning_rate,weight_decay,"
            "num_parameters,best_validation_mse,test_rmse,total_time\n"
        )

    input_size = int(config["model"].get("input_size", 8))
    output_size = int(config["model"].get("output_size", 2))

    if input_size != 8:
        raise ValueError(
            f"Energy Efficiency requires input_size=8, "
            f"but input_size={input_size} was provided in config.yaml."
        )

    if output_size != 2:
        raise ValueError(
            f"Energy Efficiency requires output_size=2 for Heating Load and Cooling Load, "
            f"but output_size={output_size} was provided in config.yaml."
        )

    hidden_sizes_values = _normalize_hidden_sizes_grid(
        config["model"].get("hidden_sizes", [[64, 32, 16]])
    )

    batch_size_values: Sequence[int] = _as_sequence(
        config["model"].get("batch_size", config.get("loaders", {}).get("batch_size", 64))
    )

    lr_values: Sequence[float] = _as_sequence(
        config["optimizer"].get("lr", 1e-3)
    )

    weight_decay = float(config["optimizer"].get("weight_decay", 0.0))

    betas = tuple(float(beta) for beta in config["optimizer"].get("betas", [0.9, 0.999]))

    activation_name = config["model"].get("activation", "relu")
    activation = get_activation(activation_name)

    use_layer_norm = bool(config["model"].get("use_layer_norm", False))
    bias = bool(config["model"].get("bias", True))

    for hidden_sizes in hidden_sizes_values:
        for batch_size in batch_size_values:
            for lr in lr_values:
                batch_size = int(batch_size)
                lr = float(lr)

                hidden_sizes_name = "-".join(map(str, hidden_sizes))

                print(
                    f"Evaluating hidden_sizes = {hidden_sizes}, "
                    f"batch_size = {batch_size}, "
                    f"lr = {lr}, "
                    f"weight_decay = {weight_decay}"
                )

                model = DNNBlock(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=output_size,
                    activation=activation,
                    use_layer_norm=use_layer_norm,
                    bias=bias,
                    device=DEVICE,
                    dtype=DTYPE,
                )

                num_parameters = model.count_trainable_parameters()

                optimizer = optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=betas,
                )

                train_loader, val_loader, test_loader = reader.make_loaders(
                    batch_size
                )

                config_folder_name = (
                    f"dnn_hidden_{hidden_sizes_name}"
                    f"_batch_size_{batch_size}"
                    f"_lr_{lr}"
                )

                config_results_folder = os.path.join(
                    rundir,
                    config_folder_name,
                )

                os.makedirs(config_results_folder, exist_ok=True)

                checkpoint_name = (
                    f"dnn_hidden_{hidden_sizes_name}"
                    f"_bs{batch_size}"
                    f"_lr{lr}"
                    f"__CKPT.pt"
                )

                val_loss, train_losses, val_losses, test_loss, elapsed_str, epoch_times = fit(
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
                        checkpoint_name,
                    ),
                    device=DEVICE,
                    dtype=DTYPE,
                    print_model_summary=True,
                    batch_size=batch_size,
                    lr=lr,
                    hidden_sizes=hidden_sizes,
                )

                print(f"Best validation MSE: {val_loss}")
                print(f"Test RMSE: {test_loss}")

                log_filename = os.path.join(
                    config_results_folder,
                    f"{config_folder_name}.txt",
                )

                with open(log_filename, "w") as log_file:
                    log_file.write(
                        f"Evaluating parameters:\n"
                        f"hidden_sizes = {hidden_sizes}\n"
                        f"batch_size = {batch_size}\n"
                        f"learning_rate = {lr}\n"
                        f"weight_decay = {weight_decay}\n"
                        f"betas = {betas}\n"
                        f"activation = {activation_name}\n"
                        f"use_layer_norm = {use_layer_norm}\n"
                        f"bias = {bias}\n"
                        f"num_parameters = {num_parameters}\n"
                    )
                    log_file.write(f"Best Validation MSE: {val_loss}\n")
                    log_file.write(f"Final Test RMSE: {test_loss}\n")
                    log_file.write(f"Training Loss per epoch: {train_losses}\n")
                    log_file.write(f"Validation Loss per epoch: {val_losses}\n")
                    log_file.write(f"Training Time (h:m:s): {elapsed_str}\n")
                    log_file.write(f"Epoch Times (seconds): {epoch_times}\n")

                with open(hyperparameter_results_filename, "a") as result_file:
                    result_file.write(
                        f"{hidden_sizes},"
                        f"{batch_size},"
                        f"{lr},"
                        f"{weight_decay},"
                        f"{num_parameters},"
                        f"{val_loss},"
                        f"{test_loss},"
                        f"{elapsed_str}\n"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_test_loss = test_loss
                    best_params = {
                        "hidden_sizes": hidden_sizes,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "num_parameters": num_parameters,
                        "folder_name": config_folder_name,
                    }

    if best_params is not None:
        best_hidden_sizes = best_params["hidden_sizes"]
        best_batch_size = best_params["batch_size"]
        best_lr = best_params["lr"]
        best_weight_decay = best_params["weight_decay"]
        best_num_parameters = best_params["num_parameters"]

        print("\nBest parameters found:")
        print(f"hidden_sizes: {best_hidden_sizes}")
        print(f"batch_size: {best_batch_size}")
        print(f"learning rate: {best_lr}")
        print(f"weight_decay: {best_weight_decay}")
        print(f"num_parameters: {best_num_parameters}")
        print(f"Best validation MSE: {best_val_loss}")
        print(f"Test RMSE for this config: {best_test_loss}")

        src_best_folder = os.path.join(
            rundir,
            best_params["folder_name"],
        )

        dst_best_folder = os.path.join(
            rundir,
            "best_hparams",
        )

        if os.path.exists(dst_best_folder):
            shutil.rmtree(dst_best_folder)

        shutil.copytree(src_best_folder, dst_best_folder)

        with open(os.path.join(dst_best_folder, "best_summary.txt"), "w") as fsum:
            fsum.write("mejor configuración de hiperparámetros\n")
            fsum.write(f"hidden_sizes: {best_hidden_sizes}\n")
            fsum.write(f"batch_size: {best_batch_size}\n")
            fsum.write(f"learning rate: {best_lr}\n")
            fsum.write(f"weight_decay: {best_weight_decay}\n")
            fsum.write(f"num_parameters: {best_num_parameters}\n")
            fsum.write(f"best validation MSE: {best_val_loss}\n")
            fsum.write(f"test RMSE (mejor config): {best_test_loss}\n")

    else:
        print("No valid configuration found during the sweep.")

