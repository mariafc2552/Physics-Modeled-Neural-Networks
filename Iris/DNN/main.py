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
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=DTYPE)
        targets = targets.to(device=device, dtype=torch.long).view(-1)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)

    mean_loss = total_loss / max(len(dataloader), 1)
    accuracy = total_correct / max(total_samples, 1)
    error_rate = 1.0 - accuracy

    return mean_loss, accuracy, error_rate


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

    if isinstance(hidden_sizes_config, int):
        return [(int(hidden_sizes_config),)]

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


def _as_bool(value: Any) -> bool:
    """
    Convert common YAML/string boolean values to bool.
    """
    if isinstance(value, bool):
        return value

    return str(value).lower() in {"true", "1", "yes", "y"}


def _get_activation(activation_name):
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

    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    valid_errors = []
    epoch_times = []

    model = model.to(**factory_kwargs)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device, dtype=torch.long).view(-1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()

        train_loss /= max(len(train_dataloader), 1)
        train_losses.append(train_loss)

        model.eval()
        valid_loss, valid_acc, valid_error = _evaluate_epoch(
            model,
            valid_dataloader,
            criterion,
            device=device,
        )

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        valid_errors.append(valid_error)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(
            f"TRAIN LOSS (CE): {train_loss:.6f} | "
            f"VALID LOSS (CE): {valid_loss:.6f} | "
            f"VALID ACC: {valid_acc:.4f} | "
            f"VALID ERROR: {valid_error:.4f} | "
            f"EPOCH TIME: {epoch_time:.2f}s"
        )

        monitored_value = -valid_acc
        stop = stopper(monitored_value, model.state_dict())

        if checkpoint_path is not None:
            stopper.save(checkpoint_path)

        if stop:
            print("\nEARLY STOPPING BASED ON VALIDATION ACCURACY\n")
            break

    if stopper.best_model is not None:
        model.load_state_dict(stopper.best_model)

    best_epoch_index = max(
        range(len(valid_accuracies)),
        key=lambda idx: (valid_accuracies[idx], -valid_losses[idx]),
    )

    best_valid_acc = valid_accuracies[best_epoch_index]
    best_valid_loss = valid_losses[best_epoch_index]
    best_valid_error = valid_errors[best_epoch_index]

    print("\nTRAINING COMPLETE\nTOTAL TIME:", longtiming(time.time() - start))
    print(f"BEST VALID ACC: {best_valid_acc:.6f}")
    print(f"VALID LOSS AT BEST VALID ACC: {best_valid_loss:.6f}")
    print(f"VALID ERROR AT BEST VALID ACC: {best_valid_error:.6f}")

    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_start = time.time()
    model.eval()

    test_loss, test_acc, test_error = _evaluate_epoch(
        model,
        test_dataloader,
        criterion,
        device=device,
    )

    test_time_s = time.time() - test_start
    test_time_str = longtiming(test_time_s)

    print("\nTEST METRICS:")
    print(f"Test CE Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_acc:.6f}")
    print(f"Test Error: {test_error:.6f}")
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

    validation_metrics_path = os.path.join(results_folder, "validation_metrics.txt")

    with open(validation_metrics_path, "w") as f:
        f.write("epoch, train_loss, valid_loss, valid_accuracy, valid_error\n")
        for idx, (tr_loss, va_loss, va_acc, va_err) in enumerate(
            zip(train_losses, valid_losses, valid_accuracies, valid_errors),
            start=1,
        ):
            f.write(f"{idx}, {tr_loss}, {va_loss}, {va_acc}, {va_err}\n")

    elapsed = time.time() - start
    elapsed_str = longtiming(elapsed)

    return (
        best_valid_acc,
        best_valid_loss,
        best_valid_error,
        train_losses,
        valid_losses,
        valid_accuracies,
        valid_errors,
        test_loss,
        test_acc,
        test_error,
        elapsed_str,
        epoch_times,
    )


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))
    baserundir = os.path.join(rootdir, "runs_dnn_iris")
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

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_val_error = 1.0
    best_test_loss = float("inf")
    best_test_acc = 0.0
    best_test_error = 1.0
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
            "activation,use_layer_norm,num_parameters,"
            "best_validation_accuracy,best_validation_loss,best_validation_error,"
            "test_loss,test_accuracy,test_error,total_time\n"
        )

    input_size = int(config["model"].get("input_size", 4))
    output_size = int(config["model"].get("output_size", 3))

    if input_size != 4:
        raise ValueError(
            f"Iris requires input_size=4, but input_size={input_size} "
            f"was provided in config.yaml."
        )

    if output_size != 3:
        raise ValueError(
            f"Iris requires output_size=3 for multi-class classification, "
            f"but output_size={output_size} was provided in config.yaml."
        )

    hidden_sizes_values = _normalize_hidden_sizes_grid(
        config["model"].get("hidden_sizes", [[64, 32, 16]])
    )

    activation_values: Sequence[str] = _as_sequence(
        config["model"].get("activation", "relu")
    )

    use_layer_norm_values: Sequence[bool] = _as_sequence(
        config["model"].get("use_layer_norm", False)
    )

    batch_size_values: Sequence[int] = _as_sequence(
        config["model"].get("batch_size", config.get("loaders", {}).get("batch_size", 64))
    )

    lr_values: Sequence[float] = _as_sequence(
        config["optimizer"].get("lr", 1e-3)
    )

    weight_decay_values: Sequence[float] = _as_sequence(
        config["optimizer"].get("weight_decay", 0.0)
    )

    betas = tuple(float(beta) for beta in config["optimizer"].get("betas", [0.9, 0.999]))

    for hidden_sizes in hidden_sizes_values:
        for activation_name in activation_values:
            for use_layer_norm in use_layer_norm_values:
                for batch_size in batch_size_values:
                    for lr in lr_values:
                        for weight_decay in weight_decay_values:
                            batch_size = int(batch_size)
                            lr = float(lr)
                            weight_decay = float(weight_decay)
                            activation_name = str(activation_name)
                            activation = _get_activation(activation_name)
                            use_layer_norm = _as_bool(use_layer_norm)

                            hidden_sizes_name = "-".join(map(str, hidden_sizes))

                            print(
                                f"Evaluating hidden_sizes = {hidden_sizes}, "
                                f"batch_size = {batch_size}, "
                                f"lr = {lr}, "
                                f"weight_decay = {weight_decay}, "
                                f"activation = {activation_name}, "
                                f"use_layer_norm = {use_layer_norm}"
                            )

                            model = DNNBlock(
                                input_size=input_size,
                                hidden_sizes=hidden_sizes,
                                output_size=output_size,
                                activation=activation,
                                use_layer_norm=use_layer_norm,
                                bias=True,
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
                                f"_wd_{weight_decay}"
                                f"_act_{activation_name}"
                                f"_ln_{use_layer_norm}"
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
                                f"_wd{weight_decay}"
                                f"_act{activation_name}"
                                f"_ln{use_layer_norm}"
                                f"__CKPT.pt"
                            )

                            (
                                val_acc,
                                val_loss,
                                val_error,
                                train_losses,
                                val_losses,
                                val_accuracies,
                                val_errors,
                                test_loss,
                                test_acc,
                                test_error,
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
                                    checkpoint_name,
                                ),
                                device=DEVICE,
                                dtype=DTYPE,
                                print_model_summary=True,
                                batch_size=batch_size,
                                lr=lr,
                                hidden_sizes=hidden_sizes,
                            )

                            print(f"Best validation accuracy: {val_acc}")
                            print(f"Validation loss at best accuracy: {val_loss}")
                            print(f"Test accuracy: {test_acc}")

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
                                    f"num_parameters = {num_parameters}\n"
                                )
                                log_file.write(f"Best Validation Accuracy: {val_acc}\n")
                                log_file.write(f"Validation Loss at Best Accuracy: {val_loss}\n")
                                log_file.write(f"Validation Error at Best Accuracy: {val_error}\n")
                                log_file.write(f"Final Test Loss: {test_loss}\n")
                                log_file.write(f"Final Test Accuracy: {test_acc}\n")
                                log_file.write(f"Final Test Error: {test_error}\n")
                                log_file.write(f"Training Loss per epoch: {train_losses}\n")
                                log_file.write(f"Validation Loss per epoch: {val_losses}\n")
                                log_file.write(f"Validation Accuracy per epoch: {val_accuracies}\n")
                                log_file.write(f"Validation Error per epoch: {val_errors}\n")
                                log_file.write(f"Training Time (h:m:s): {elapsed_str}\n")
                                log_file.write(f"Epoch Times (seconds): {epoch_times}\n")

                            with open(hyperparameter_results_filename, "a") as result_file:
                                result_file.write(
                                    f"{hidden_sizes},"
                                    f"{batch_size},"
                                    f"{lr},"
                                    f"{weight_decay},"
                                    f"{activation_name},"
                                    f"{use_layer_norm},"
                                    f"{num_parameters},"
                                    f"{val_acc},"
                                    f"{val_loss},"
                                    f"{val_error},"
                                    f"{test_loss},"
                                    f"{test_acc},"
                                    f"{test_error},"
                                    f"{elapsed_str}\n"
                                )

                            if (val_acc > best_val_acc) or (
                                val_acc == best_val_acc and val_loss < best_val_loss
                            ):
                                best_val_acc = val_acc
                                best_val_loss = val_loss
                                best_val_error = val_error
                                best_test_loss = test_loss
                                best_test_acc = test_acc
                                best_test_error = test_error
                                best_params = {
                                    "hidden_sizes": hidden_sizes,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "weight_decay": weight_decay,
                                    "activation": activation_name,
                                    "use_layer_norm": use_layer_norm,
                                    "num_parameters": num_parameters,
                                    "folder_name": config_folder_name,
                                }

    if best_params is not None:
        best_hidden_sizes = best_params["hidden_sizes"]
        best_batch_size = best_params["batch_size"]
        best_lr = best_params["lr"]
        best_weight_decay = best_params["weight_decay"]
        best_activation = best_params["activation"]
        best_use_layer_norm = best_params["use_layer_norm"]
        best_num_parameters = best_params["num_parameters"]

        print("\nBest parameters found:")
        print(f"hidden_sizes: {best_hidden_sizes}")
        print(f"batch_size: {best_batch_size}")
        print(f"learning rate: {best_lr}")
        print(f"weight_decay: {best_weight_decay}")
        print(f"activation: {best_activation}")
        print(f"use_layer_norm: {best_use_layer_norm}")
        print(f"num_parameters: {best_num_parameters}")
        print(f"Best validation accuracy: {best_val_acc}")
        print(f"Validation loss at best accuracy: {best_val_loss}")
        print(f"Validation error at best accuracy: {best_val_error}")
        print(f"Test loss for this config: {best_test_loss}")
        print(f"Test accuracy for this config: {best_test_acc}")
        print(f"Test error for this config: {best_test_error}")

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
            fsum.write(f"activation: {best_activation}\n")
            fsum.write(f"use_layer_norm: {best_use_layer_norm}\n")
            fsum.write(f"num_parameters: {best_num_parameters}\n")
            fsum.write(f"best validation accuracy: {best_val_acc}\n")
            fsum.write(f"validation loss at best accuracy: {best_val_loss}\n")
            fsum.write(f"validation error at best accuracy: {best_val_error}\n")
            fsum.write(f"test loss (mejor config): {best_test_loss}\n")
            fsum.write(f"test accuracy (mejor config): {best_test_acc}\n")
            fsum.write(f"test error (mejor config): {best_test_error}\n")

    else:
        print("No valid configuration found during the sweep.")
