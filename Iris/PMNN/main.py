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


def as_bool(value):
    """
    Convert common YAML/string boolean values to bool.
    """
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


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
    Evaluate one epoch for Iris multiclass classification.

    The model returns logits of shape:

        (batch_size, 3)

    The targets are integer class labels:

        0, 1 or 2

    Therefore, the loss is CrossEntropyLoss.
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device=device, dtype=torch.long).view(-1)

        outputs = model(inputs)

        if outputs.dim() != 2:
            raise ValueError(
                f"Expected outputs with shape (batch_size, num_classes), "
                f"but received {tuple(outputs.shape)}."
            )

        if outputs.size(0) != targets.size(0):
            raise ValueError(
                f"Batch mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        if targets.numel() > 0:
            min_target = int(targets.min().item())
            max_target = int(targets.max().item())

            if min_target < 0 or max_target >= outputs.size(1):
                raise ValueError(
                    f"Invalid target labels. Expected labels in [0, {outputs.size(1) - 1}], "
                    f"but got min={min_target}, max={max_target}."
                )

        loss = criterion(outputs, targets)
        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += targets.size(0)

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
    t_step=None,
    t_end=None,
    batch_size=None,
    lr=None,
):
    """
    Train, validate and test a FitzHugh--Nagumo PMNN for Iris multiclass
    classification.

    The monitored validation metric is accuracy. Since the existing EarlyStopper
    is assumed to minimize a monitored value, we monitor -valid_acc.
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

            if outputs.dim() != 2:
                raise ValueError(
                    f"Expected outputs with shape (batch_size, num_classes), "
                    f"but received {tuple(outputs.shape)}."
                )

            if outputs.size(0) != targets.size(0):
                raise ValueError(
                    f"Batch mismatch: outputs {outputs.shape} vs targets {targets.shape}"
                )

            if targets.numel() > 0:
                min_target = int(targets.min().item())
                max_target = int(targets.max().item())

                if min_target < 0 or max_target >= outputs.size(1):
                    raise ValueError(
                        f"Invalid target labels. Expected labels in [0, {outputs.size(1) - 1}], "
                        f"but got min={min_target}, max={max_target}."
                    )

            loss = criterion(outputs, targets)

            loss.backward()

            # Gradient clipping improves stability when the FHN block is used.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        dtype=dtype,
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

    with open(os.path.join(rootdir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    results_base = config["paths"].get(
        "results",
        "runs_pmnn_iris",
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
            "Test CE Loss, Test Accuracy, Test Error, "
            "Time Step, End Time, Batch Size, LR, "
            "Input Size, Hidden Size, Output Size, "
            "Initial State Scale, State Clip, Use Layer Norm\n"
        )

    input_size = int(config["model"].get("input_size", 4))
    hidden_size = int(config["model"].get("hidden_size", 2))
    output_size = int(config["model"].get("output_size", 3))

    if input_size != 4:
        print(
            f"Warning: Iris dataset usually has input_size=4, "
            f"but input_size={input_size} was provided."
        )

    if output_size != 3:
        print(
            f"Warning: Iris dataset usually has output_size=3, "
            f"but output_size={output_size} was provided."
        )

    t_step_values: Sequence[float] = as_list(config["model"].get("t_step", [0.05]))
    t_end_values: Sequence[float] = as_list(config["model"].get("t_end", [1.0]))

    batch_size_values: Sequence[int] = as_list(
        config["model"].get(
            "batch_size",
            config.get("loaders", {}).get("batch_size", 32),
        )
    )

    lr_values: Sequence[float] = as_list(config["optimizer"]["lr"])

    a = float(config["model"].get("a", 0.2))
    b = float(config["model"].get("b", 0.02))
    g = float(config["model"].get("g", 3.0))
    I = float(config["model"].get("I", 0.0))

    initial_state_scale = float(config["model"].get("initial_state_scale", 1.0))
    state_clip = config["model"].get("state_clip", 20.0)

    if state_clip is not None:
        state_clip = float(state_clip)

    use_layer_norm = as_bool(config["model"].get("use_layer_norm", True))

    optimizer_name = str(config["optimizer"].get("name", "Adam")).lower()
    weight_decay = float(config["optimizer"].get("weight_decay", 0.0))
    momentum = float(config["optimizer"].get("momentum", 0.9))
    betas = tuple(config["optimizer"].get("betas", [0.9, 0.999]))

    for t_step in t_step_values:
        for t_end in t_end_values:
            for batch_size in batch_size_values:
                for lr in lr_values:
                    t_step = float(t_step)
                    t_end = float(t_end)
                    batch_size = int(batch_size)
                    lr = float(lr)

                    if t_step <= 0.0 or t_end <= 0.0 or t_step > t_end:
                        print(
                            f"Invalid configuration: "
                            f"t_step = {t_step}, t_end = {t_end}. "
                            f"Ignoring configuration..."
                        )
                        continue

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
                        activation=nn.SiLU,
                        dt=t_step,
                        t_end=t_end,
                        a=a,
                        b=b,
                        g=g,
                        I=I,
                        initial_state_scale=initial_state_scale,
                        state_clip=state_clip,
                        use_layer_norm=use_layer_norm,
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
                        f"t_step_{t_step}"
                        f"_t_end_{t_end}"
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
                            f"pmnn_{config_name}__CKPT.pt",
                        ),
                        device=DEVICE,
                        dtype=DTYPE,
                        print_model_summary=True,
                        t_step=t_step,
                        t_end=t_end,
                        batch_size=batch_size,
                        lr=lr,
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
                            f"t_step = {t_step}, "
                            f"t_end = {t_end}, "
                            f"batch_size = {batch_size}, "
                            f"lr = {lr}, "
                            f"input_size = {input_size}, "
                            f"hidden_size = {hidden_size}, "
                            f"output_size = {output_size}, "
                            f"initial_state_scale = {initial_state_scale}, "
                            f"state_clip = {state_clip}, "
                            f"use_layer_norm = {use_layer_norm}\n"
                        )
                        log_file.write(f"Best Validation Accuracy: {val_acc}\n")
                        log_file.write(f"Validation Loss at Best Accuracy: {val_loss}\n")
                        log_file.write(f"Validation Error at Best Accuracy: {val_error}\n")
                        log_file.write(f"Final Test CE Loss: {test_loss}\n")
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
                            f"{t_step}, {t_end}, {batch_size}, {lr}, "
                            f"{input_size}, {hidden_size}, {output_size}, "
                            f"{initial_state_scale}, {state_clip}, {use_layer_norm}\n"
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
                            t_step,
                            t_end,
                            batch_size,
                            lr,
                            input_size,
                            hidden_size,
                            output_size,
                            initial_state_scale,
                            state_clip,
                            use_layer_norm,
                        )

    if best_params is not None:
        (
            best_t_step,
            best_t_end,
            best_batch_size,
            best_lr,
            best_input_size,
            best_hidden_size,
            best_output_size,
            best_initial_state_scale,
            best_state_clip,
            best_use_layer_norm,
        ) = best_params

        print(f"\nBest parameters found based on validation accuracy:")
        print(f"t_step: {best_t_step}")
        print(f"t_end: {best_t_end}")
        print(f"batch_size: {best_batch_size}")
        print(f"learning rate: {best_lr}")
        print(f"input_size: {best_input_size}")
        print(f"hidden_size: {best_hidden_size}")
        print(f"output_size: {best_output_size}")
        print(f"initial_state_scale: {best_initial_state_scale}")
        print(f"state_clip: {best_state_clip}")
        print(f"use_layer_norm: {best_use_layer_norm}")
        print(f"Best validation_accuracy: {best_val_acc}")
        print(f"Validation loss at best accuracy: {best_val_loss}")
        print(f"Validation error at best accuracy: {best_val_error}")
        print(f"Test loss for this config: {best_test_loss}")
        print(f"Test accuracy for this config: {best_test_acc}")
        print(f"Test error for this config: {best_test_error}")

        best_config_foldername = (
            f"t_step_{best_t_step}"
            f"_t_end_{best_t_end}"
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
            fsum.write(f"t_step: {best_t_step}\n")
            fsum.write(f"t_end: {best_t_end}\n")
            fsum.write(f"batch_size: {best_batch_size}\n")
            fsum.write(f"learning rate: {best_lr}\n")
            fsum.write(f"input_size: {best_input_size}\n")
            fsum.write(f"hidden_size: {best_hidden_size}\n")
            fsum.write(f"output_size: {best_output_size}\n")
            fsum.write(f"initial_state_scale: {best_initial_state_scale}\n")
            fsum.write(f"state_clip: {best_state_clip}\n")
            fsum.write(f"use_layer_norm: {best_use_layer_norm}\n")
            fsum.write(f"best validation_accuracy: {best_val_acc}\n")
            fsum.write(f"validation loss at best accuracy: {best_val_loss}\n")
            fsum.write(f"validation error at best accuracy: {best_val_error}\n")
            fsum.write(f"test loss (best config): {best_test_loss}\n")
            fsum.write(f"test accuracy (best config): {best_test_acc}\n")
            fsum.write(f"test error (best config): {best_test_error}\n")

    else:
        print("No valid configuration found during the sweep.")

