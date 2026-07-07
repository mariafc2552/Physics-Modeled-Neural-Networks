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


def as_bool(value: Any) -> bool:
    """
    Convert common YAML/string boolean values to bool.
    """
    value = unwrap_scalar(value)

    if isinstance(value, bool):
        return value

    return str(value).lower() in {"true", "1", "yes", "y"}


def normalize_hidden_sizes_grid(value):
    """
    Normalize hidden_sizes for grid search.

    Accepted formats:

        hidden_sizes: [64, 32, 16]

    or:

        hidden_sizes:
          - [64, 32, 16]
          - [128, 64, 32]

    or:

        hidden_sizes: 64

    If the config does not provide hidden_sizes, the default architecture is:

        (64, 32, 16)
    """
    if value is None:
        return [(64, 32, 16)]

    if isinstance(value, int):
        return [(int(value),)]

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("hidden_sizes cannot be empty.")

        if all(isinstance(v, int) for v in value):
            return [tuple(int(v) for v in value)]

        if all(isinstance(v, (list, tuple)) for v in value):
            return [tuple(int(h) for h in architecture) for architecture in value]

    raise ValueError(
        "Invalid hidden_sizes format. Use [64, 32, 16] or "
        "[[64, 32, 16], [128, 64, 32]]."
    )


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


def tag_value(value):
    """
    Create a safe string tag for folder/file names.
    """
    if isinstance(value, (list, tuple)):
        return "-".join(str(v).replace(".", "p") for v in value)

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
    Evaluate one epoch for Breast Cancer binary classification.

    Model output:
        outputs.shape = (batch_size, 1)

    Targets:
        targets.shape = (batch_size, 1)
        values in {0.0, 1.0}

    Loss:
        BCEWithLogitsLoss
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=dtype)

        targets_float = targets.to(device=device, dtype=torch.float32).view(-1, 1)
        targets_int = targets.to(device=device, dtype=torch.long).view(-1)

        outputs = model(inputs)

        if outputs.dim() == 1:
            outputs = outputs.view(-1, 1)

        if outputs.shape != targets_float.shape:
            raise ValueError(
                f"Shape mismatch: outputs {outputs.shape} vs targets {targets_float.shape}"
            )

        loss = criterion(outputs, targets_float)
        total_loss += loss.item()

        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).long().view(-1)

        total_correct += (predictions == targets_int).sum().item()
        total_samples += targets_int.size(0)

    mean_loss = total_loss / max(len(dataloader), 1)
    accuracy = total_correct / max(total_samples, 1)
    error_rate = 1.0 - accuracy

    return mean_loss, accuracy, error_rate


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
    use_layer_norm=None,
    activation=None,
):
    """
    Train, validate and test a DNN model for Breast Cancer binary classification.

    The model returns one raw logit per sample and is trained with
    BCEWithLogitsLoss.

    The selected validation metric is accuracy. Since EarlyStopper minimizes
    the monitored value, we pass -valid_acc.
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

    criterion = nn.BCEWithLogitsLoss()
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
            targets = targets.to(device=device, dtype=torch.float32).view(-1, 1)

            outputs = model(inputs)

            if outputs.dim() == 1:
                outputs = outputs.view(-1, 1)

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
        valid_loss, valid_acc, valid_error = _evaluate_epoch(
            model,
            valid_dataloader,
            criterion,
            device=device,
            dtype=dtype,
        )

        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        valid_errors.append(valid_error)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(
            f"TRAIN LOSS (BCE): {train_loss:.6f} | "
            f"VALID LOSS (BCE): {valid_loss:.6f} | "
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
        dtype=dtype,
    )

    test_time_s = time.time() - test_start
    test_time_str = longtiming(test_time_s)

    print("\nTEST METRICS:")
    print(f"Test BCE Loss: {test_loss:.6f}")
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
    baserundir = os.path.join(rootdir, "runs_dnn_breast_cancer")
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
            "File Name, Best Valid Accuracy, Best Valid Loss, Best Valid Error, "
            "Test BCE Loss, Test Accuracy, Test Error, "
            "Hidden Sizes, Layer Norm, Activation, "
            "Batch Size, LR, Input Size, Output Size\n"
        )

    input_size = int(unwrap_scalar(config["model"].get("input_size", 30)))
    output_size = int(unwrap_scalar(config["model"].get("output_size", 1)))

    if input_size != 30:
        print(
            f"Warning: Breast Cancer normally has input_size=30, "
            f"but input_size={input_size} was provided."
        )

    if output_size != 1:
        print(
            f"Warning: Breast Cancer binary classification with BCEWithLogitsLoss "
            f"normally has output_size=1, but output_size={output_size} was provided."
        )

    hidden_sizes_values = normalize_hidden_sizes_grid(
        config["model"].get("hidden_sizes", [64, 32, 16])
    )

    use_layer_norm_values: Sequence[bool] = as_list(
        config["model"].get("use_layer_norm", [False])
    )

    activation_values: Sequence[str] = as_list(
        config["model"].get("activation", ["relu"])
    )

    batch_size_values: Sequence[int] = as_list(
        config["model"].get(
            "batch_size",
            config.get("loaders", {}).get("batch_size", 32),
        )
    )

    lr_values: Sequence[float] = as_list(config["optimizer"]["lr"])

    optimizer_name = str(config["optimizer"].get("name", "Adam")).lower()
    weight_decay = float(config["optimizer"].get("weight_decay", 0.0))
    momentum = float(config["optimizer"].get("momentum", 0.9))
    betas = tuple(config["optimizer"].get("betas", [0.9, 0.999]))

    for hidden_sizes in hidden_sizes_values:
        for use_layer_norm in use_layer_norm_values:
            for activation_name in activation_values:
                for batch_size in batch_size_values:
                    for lr in lr_values:
                        hidden_sizes = tuple(int(h) for h in hidden_sizes)
                        use_layer_norm = as_bool(use_layer_norm)
                        activation_name = str(activation_name)
                        activation = get_activation(activation_name)
                        batch_size = int(batch_size)
                        lr = float(lr)

                        print(
                            f"Evaluating hidden_sizes = {hidden_sizes}, "
                            f"use_layer_norm = {use_layer_norm}, "
                            f"activation = {activation_name}, "
                            f"batch_size = {batch_size}, "
                            f"lr = {lr}, "
                            f"input_size = {input_size}, "
                            f"output_size = {output_size}"
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

                        hidden_tag = tag_value(hidden_sizes)

                        config_name = (
                            f"dnn_hidden_{hidden_tag}"
                            f"_ln_{use_layer_norm}"
                            f"_act_{activation_name}"
                            f"_batch_size_{batch_size}"
                            f"_lr_{lr}"
                        )

                        config_results_folder = os.path.join(rundir, config_name)
                        os.makedirs(config_results_folder, exist_ok=True)

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
                                f"{config_name}__CKPT.pt",
                            ),
                            device=DEVICE,
                            dtype=DTYPE,
                            print_model_summary=True,
                            batch_size=batch_size,
                            lr=lr,
                            hidden_sizes=hidden_sizes,
                            use_layer_norm=use_layer_norm,
                            activation=activation_name,
                        )

                        print(f"Validation Accuracy: {val_acc}")
                        print(f"Validation Loss at Best Accuracy: {val_loss}")

                        log_filename = os.path.join(
                            config_results_folder,
                            f"{config_name}.txt",
                        )

                        with open(log_filename, "w") as log_file:
                            log_file.write(
                                f"Evaluating parameters: "
                                f"hidden_sizes = {hidden_sizes}, "
                                f"use_layer_norm = {use_layer_norm}, "
                                f"activation = {activation_name}, "
                                f"batch_size = {batch_size}, "
                                f"lr = {lr}, "
                                f"input_size = {input_size}, "
                                f"output_size = {output_size}\n"
                            )
                            log_file.write(f"Best Validation Accuracy: {val_acc}\n")
                            log_file.write(f"Validation Loss at Best Accuracy: {val_loss}\n")
                            log_file.write(f"Validation Error at Best Accuracy: {val_error}\n")
                            log_file.write(f"Final Test BCE Loss: {test_loss}\n")
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
                                f"{config_name}.txt, "
                                f"{val_acc}, {val_loss}, {val_error}, "
                                f"{test_loss}, {test_acc}, {test_error}, "
                                f"{hidden_sizes}, {use_layer_norm}, "
                                f"{activation_name}, {batch_size}, {lr}, "
                                f"{input_size}, {output_size}\n"
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
                            best_params = (
                                hidden_sizes,
                                use_layer_norm,
                                activation_name,
                                batch_size,
                                lr,
                                input_size,
                                output_size,
                            )

    if best_params is not None:
        (
            best_hidden_sizes,
            best_use_layer_norm,
            best_activation,
            best_batch_size,
            best_lr,
            best_input_size,
            best_output_size,
        ) = best_params

        print(f"\nBest parameters found based on validation accuracy:")
        print(f"hidden_sizes: {best_hidden_sizes}")
        print(f"use_layer_norm: {best_use_layer_norm}")
        print(f"activation: {best_activation}")
        print(f"batch_size: {best_batch_size}")
        print(f"learning rate: {best_lr}")
        print(f"input_size: {best_input_size}")
        print(f"output_size: {best_output_size}")
        print(f"Best validation_accuracy: {best_val_acc}")
        print(f"Validation loss at best accuracy: {best_val_loss}")
        print(f"Validation error at best accuracy: {best_val_error}")
        print(f"Test loss for this config: {best_test_loss}")
        print(f"Test accuracy for this config: {best_test_acc}")
        print(f"Test error for this config: {best_test_error}")

        best_config_foldername = (
            f"dnn_hidden_{tag_value(best_hidden_sizes)}"
            f"_ln_{best_use_layer_norm}"
            f"_act_{best_activation}"
            f"_batch_size_{best_batch_size}"
            f"_lr_{best_lr}"
        )

        src_best_folder = os.path.join(rundir, best_config_foldername)
        dst_best_folder = os.path.join(rundir, "best_hparams")

        if os.path.exists(dst_best_folder):
            shutil.rmtree(dst_best_folder)

        shutil.copytree(src_best_folder, dst_best_folder)

        with open(os.path.join(dst_best_folder, "best_summary.txt"), "w") as fsum:
            fsum.write("Best hyperparameters found based on validation accuracy:\n")
            fsum.write(f"hidden_sizes: {best_hidden_sizes}\n")
            fsum.write(f"use_layer_norm: {best_use_layer_norm}\n")
            fsum.write(f"activation: {best_activation}\n")
            fsum.write(f"batch_size: {best_batch_size}\n")
            fsum.write(f"learning rate: {best_lr}\n")
            fsum.write(f"input_size: {best_input_size}\n")
            fsum.write(f"output_size: {best_output_size}\n")
            fsum.write(f"best validation_accuracy: {best_val_acc}\n")
            fsum.write(f"validation loss at best accuracy: {best_val_loss}\n")
            fsum.write(f"validation error at best accuracy: {best_val_error}\n")
            fsum.write(f"test loss (best config): {best_test_loss}\n")
            fsum.write(f"test accuracy (best config): {best_test_acc}\n")
            fsum.write(f"test error (best config): {best_test_error}\n")

    else:
        print("No valid configuration found during the sweep.")

