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
from src.NODE import NODEblock

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@torch.no_grad()
def _evaluate_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    criterion: nn.Module,
                    *,
                    device: torch.device):
    total = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(device=device, dtype=DTYPE)
        targets = targets.to(device=device, dtype=torch.float32)

        outputs = model(inputs)

        if targets.dim() == 1:
            targets = targets.view(outputs.shape)

        if outputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        loss = criterion(outputs, targets)
        total += loss.item()

    return total / max(len(dataloader), 1)


def fit(model,
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
        hidden_dim=None,
        num_layer=None):

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
            targets = targets.to(device=device, dtype=torch.float32)

            outputs = model(inputs)

            if targets.dim() == 1:
                targets = targets.view(outputs.shape)

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
        valid_loss = _evaluate_epoch(model, valid_dataloader, criterion, device=device)
        valid_losses.append(valid_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(f"TRAIN LOSS (MSE): {train_loss:.6f} | VALID LOSS (MSE): {valid_loss:.6f} | EPOCH TIME: {epoch_time:.2f}s")

        stop = stopper(valid_loss, model.state_dict())
        if checkpoint_path is not None:
            stopper.save(checkpoint_path)
        if stop:
            print("\nEARLY STOPPING\n")
            break

    if stopper.best_model is not None:
        model.load_state_dict(stopper.best_model)

    best_valid_loss = min(valid_losses) if len(valid_losses) > 0 else float("inf")

    print("\nTRAINING COMPLETE\nTOTAL TIME:", longtiming(time.time() - start))
    print(f"BEST VALID LOSS (MSE): {best_valid_loss:.6f}")

    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_start = time.time()
    model.eval()

    test_mse = _evaluate_epoch(model, test_dataloader, criterion, device=device)
    test_rmse = test_mse ** 0.5

    test_time_s = time.time() - test_start
    test_time_str = longtiming(test_time_s)

    print("\nTEST METRICS:")
    print(f"MSE Test Loss: {test_mse:.6f}")
    print(f"RMSE Test Loss: {test_rmse:.6f}")
    print(f"TEST INFERENCE TIME: {test_time_s:.2f}s ({test_time_str})")

    os.makedirs(results_folder, exist_ok=True)

    best_model_path = os.path.join(results_folder, "best_model_state_dict.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(best_model_path)}\n")
    logger.info(f"Model saved to: {os.path.abspath(best_model_path)}")

    save_loss_curves(train_losses, valid_losses, batch_size, lr, hidden_dim, num_layer, results_folder)

    validation_metrics_path = os.path.join(results_folder, "validation_metrics.txt")

    with open(validation_metrics_path, "w") as f:
        f.write("epoch, train_loss, valid_loss\n")
        for idx, (tr_loss, va_loss) in enumerate(
            zip(train_losses, valid_losses),
            start=1,
        ):
            f.write(f"{idx}, {tr_loss}, {va_loss}\n")

    elapsed = time.time() - start
    elapsed_str = longtiming(elapsed)

    return best_valid_loss, train_losses, valid_losses, test_mse, test_rmse, elapsed_str, epoch_times


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))
    baserundir = os.path.join(rootdir, "runs_node_energy_efficiency")
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

    logging.basicConfig(filename=os.path.join(rundir, "runinfo.log"), level=logging.INFO)

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

    best_val_loss = float('inf')
    best_test_mse = float('inf')
    best_test_rmse = float('inf')
    best_params = None

    results_folder = rundir
    os.makedirs(results_folder, exist_ok=True)
    hyperparameter_results_filename = os.path.join(results_folder, "hyperparameter_results.txt")
    with open(hyperparameter_results_filename, 'w') as result_file:
        result_file.write(
            "File Name, Best Valid MSE, Test MSE, Test RMSE, "
            "Batch Size, LR, Num Layer, Hidden Dim, Input Size, Output Size\n"
        )

    input_size = int(config["model"].get("input_size", 8))
    output_size = int(config["model"].get("output_size", 2))

    if input_size != 8:
        raise ValueError(
            f"Energy Efficiency requires input_size=8, but input_size={input_size} "
            f"was provided in config.yaml."
        )

    if output_size != 2:
        raise ValueError(
            f"Energy Efficiency requires output_size=2, but output_size={output_size} "
            f"was provided in config.yaml."
        )

    batch_size_values: Sequence[int] = config["model"]["batch_size"]
    lr_values: Sequence[float] = config["optimizer"]["lr"]
    num_layers: Sequence[int] = config["model"]["num_layers"]
    hidden_dims: Sequence[int] = config["model"]["hidden_dims"]
    weight_decay = float(config["optimizer"].get("weight_decay", 0.0))
    betas = tuple(config["optimizer"].get("betas", [0.9, 0.999]))

    for layer in num_layers:
        for hidden_dimension in hidden_dims:
            for batch_size in batch_size_values:
                for lr in lr_values:

                    batch_size = int(batch_size)
                    lr = float(lr)
                    layer = int(layer)
                    hidden_dimension = int(hidden_dimension)

                    print(
                        f"Evaluating batch_size = {batch_size}, "
                        f"lr = {lr}, "
                        f"num_layer = {layer}, "
                        f"hidden_dim = {hidden_dimension}, "
                        f"input_size = {input_size}, "
                        f"output_size = {output_size}"
                    )

                    model = NODEblock(
                        input_size=input_size,
                        output_size=output_size,
                        hidden_dim=hidden_dimension,
                        num_layers=layer,
                    )
                    model = model.to(DEVICE)

                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=lr,
                        weight_decay=weight_decay,
                        betas=betas,
                    )

                    train_loader, val_loader, test_loader = reader.make_loaders(batch_size)

                    config_results_folder = os.path.join(
                        rundir,
                        f"batch_size_{batch_size}_lr_{lr}_num_layer_{layer}_hidden_dim_{hidden_dimension}"
                    )
                    os.makedirs(config_results_folder, exist_ok=True)

                    (
                        val_loss,
                        train_losses,
                        val_losses,
                        test_mse,
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
                            f"batch_size_{batch_size}_lr_{lr}_num_layer_{layer}_hidden_dim_{hidden_dimension}__CKPT.pt"
                        ),
                        device=DEVICE,
                        dtype=DTYPE,
                        print_model_summary=True,
                        batch_size=batch_size,
                        lr=lr,
                        hidden_dim=hidden_dimension,
                        num_layer=layer
                    )

                    print(f"Validation Loss: {val_loss}")
                    print(f"Test MSE: {test_mse}")
                    print(f"Test RMSE: {test_rmse}")

                    log_filename = os.path.join(
                        config_results_folder,
                        f"batch_size_{batch_size}_lr_{lr}_num_layer_{layer}_hidden_dim_{hidden_dimension}.txt"
                    )
                    with open(log_filename, 'w') as log_file:
                        log_file.write(
                            f"Evaluating parameters: "
                            f"batch_size = {batch_size}, "
                            f"lr = {lr}, "
                            f"num_layer = {layer}, "
                            f"hidden_dim = {hidden_dimension}, "
                            f"input_size = {input_size}, "
                            f"output_size = {output_size}\n"
                        )
                        log_file.write(f"Best Validation MSE: {val_loss}\n")
                        log_file.write(f"Final Test MSE: {test_mse}\n")
                        log_file.write(f"Final Test RMSE: {test_rmse}\n")
                        log_file.write(f"Training Loss per epoch: {train_losses}\n")
                        log_file.write(f"Validation Loss per epoch: {val_losses}\n")
                        log_file.write(f"Training Time (h:m:s): {elapsed_str}\n")
                        log_file.write(f"Epoch Times (seconds): {epoch_times}\n")

                    with open(hyperparameter_results_filename, 'a') as result_file:
                        result_file.write(
                            f"batch_size_{batch_size}_lr_{lr}_num_layer_{layer}_hidden_dim_{hidden_dimension}.txt, "
                            f"{val_loss}, {test_mse}, {test_rmse}, "
                            f"{batch_size}, {lr}, {layer}, {hidden_dimension}, "
                            f"{input_size}, {output_size}\n"
                        )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_test_mse = test_mse
                        best_test_rmse = test_rmse
                        best_params = (
                            batch_size,
                            lr,
                            layer,
                            hidden_dimension,
                            input_size,
                            output_size,
                        )

    if best_params is not None:
        print(f"\nBest parameters found:")
        print(f"batch_size: {best_params[0]}")
        print(f"learning rate: {best_params[1]}")
        print(f"num layer: {best_params[2]}")
        print(f"hidden dim: {best_params[3]}")
        print(f"input_size: {best_params[4]}")
        print(f"output_size: {best_params[5]}")
        print(f"Best validation_loss: {best_val_loss}")
        print(f"Test MSE for this config: {best_test_mse}")
        print(f"Test RMSE for this config: {best_test_rmse}")

        best_config_foldername = (
            f"batch_size_{best_params[0]}"
            f"_lr_{best_params[1]}"
            f"_num_layer_{best_params[2]}"
            f"_hidden_dim_{best_params[3]}"
        )
        src_best_folder = os.path.join(rundir, best_config_foldername)
        dst_best_folder = os.path.join(rundir, "best_hparams")

        if os.path.exists(dst_best_folder):
            shutil.rmtree(dst_best_folder)
        shutil.copytree(src_best_folder, dst_best_folder)

        with open(os.path.join(dst_best_folder, "best_summary.txt"), "w") as fsum:
            fsum.write("mejor configuración de hiperparámetros\n")
            fsum.write(f"batch_size: {best_params[0]}\n")
            fsum.write(f"learning rate: {best_params[1]}\n")
            fsum.write(f"num layer: {best_params[2]}\n")
            fsum.write(f"hidden dim: {best_params[3]}\n")
            fsum.write(f"input_size: {best_params[4]}\n")
            fsum.write(f"output_size: {best_params[5]}\n")
            fsum.write(f"best validation_loss: {best_val_loss}\n")
            fsum.write(f"test MSE (mejor config): {best_test_mse}\n")
            fsum.write(f"test RMSE (mejor config): {best_test_rmse}\n")
    else:
        print("No valid configuration found during the sweep.")

