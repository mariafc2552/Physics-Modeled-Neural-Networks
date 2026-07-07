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
from src.CFC import CFCblock

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
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        if outputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}"
            )

        loss = criterion(outputs, targets)
        total += loss.item()

    return total / len(dataloader)

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
        num_units=None,
        batch_size=None,
        lr=None):

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
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
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

    print("\nTRAINING COMPLETE\nTOTAL TIME:", longtiming(time.time() - start))

    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_start = time.time()
    model.eval()
    test_loss = _evaluate_epoch(model, test_dataloader, criterion, device=device)
    test_loss = test_loss ** 0.5
    test_time_s = time.time() - test_start
    test_time_str = longtiming(test_time_s)

    print("\nTEST METRICS:")
    print(f"RMSE Test Loss: {test_loss:.6f}")
    print(f"TEST INFERENCE TIME: {test_time_s:.2f}s ({test_time_str})")

    os.makedirs(results_folder, exist_ok=True)

    best_model_path = os.path.join(results_folder, "best_model_state_dict.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(best_model_path)}\n")
    logger.info(f"Model saved to: {os.path.abspath(best_model_path)}")

    save_loss_curves(train_losses, valid_losses, num_units, batch_size, lr, results_folder)

    elapsed = time.time() - start
    elapsed_str = longtiming(elapsed)

    return valid_losses[-1], train_losses, valid_losses, test_loss, elapsed_str, epoch_times


if __name__ == "__main__":
    rootdir = os.path.dirname(os.path.realpath(__file__))
    baserundir = os.path.join(rootdir, "runs_cfc")
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

    datadir = config["paths"]["datadir"]
    reader = Reader(datadir)
    train_dataset = reader.make_dataset("train")
    val_dataset = reader.make_dataset("valid")
    test_dataset = reader.make_dataset("test")

    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_params = None

    results_folder = rundir
    os.makedirs(results_folder, exist_ok=True)
    hyperparameter_results_filename = os.path.join(results_folder, "hyperparameter_results.txt")
    with open(hyperparameter_results_filename, 'w') as result_file:
        result_file.write("File Name, Test Loss\n")

    batch_size_values: Sequence[int] = config["model"]["batch_size"]
    lr_values: Sequence[float] = config["optimizer"]["lr"]
    weight_decay = float(config["optimizer"].get("weight_decay", 0.0))
    betas = tuple(config["optimizer"].get("betas", [0.9, 0.999]))
    num_units_values: Sequence[int] = config["model"]["num_units_values"]

    for num_units in num_units_values:
        for batch_size in batch_size_values:
            for lr in lr_values:

                batch_size = int(batch_size)
                lr = float(lr)

                print(f"Evaluating batch_size = {batch_size}, lr = {lr}, num_units = {num_units_values}")

                model = CFCblock(input_size=8, num_units=num_units, output_size=1)
                model = model.to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

                train_loader, val_loader, test_loader = reader.make_loaders(batch_size)

                config_results_folder = os.path.join(
                    rundir,
                    f"num_units_{num_units}_batch_size_{batch_size}_lr_{lr}"
                )
                os.makedirs(config_results_folder, exist_ok=True)

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
                        f"cfc_neuron_{num_units}_bs{batch_size}_lr{lr}__CKPT.pt"
                    ),
                    device=DEVICE,
                    dtype=DTYPE,
                    print_model_summary=True,
                    num_units=num_units,
                    batch_size=batch_size,
                    lr=lr,
                )

                print(f"Validation Loss: {val_loss}")

                log_filename = os.path.join(
                    config_results_folder,  # <-- escribir log de la configuración en su carpeta
                    f"num_units_{num_units}_batch_size_{batch_size}_lr_{lr}.txt"
                )
                with open(log_filename, 'w') as log_file:
                    log_file.write(
                        f"Evaluating parameters: num_units = {num_units}, batch_size = {batch_size}, lr = {lr}\n"
                    )
                    log_file.write(f"Final Test Loss: {test_loss}\n")
                    log_file.write(f"Training Loss per epoch: {train_losses}\n")
                    log_file.write(f"Validation Loss per epoch: {val_losses}\n")
                    log_file.write(f"Training Time (h:m:s): {elapsed_str}\n")
                    log_file.write(f"Epoch Times (seconds): {epoch_times}\n")

                with open(hyperparameter_results_filename, 'a') as result_file:
                    result_file.write(
                        f"num_units_{num_units}_batch_size_{batch_size}_lr_{lr}.txt, {test_loss}\n"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_test_loss = test_loss
                    best_params = (num_units, batch_size, lr)

    if best_params is not None:
        print(f"\nBest parameters found:")
        print(f"num_units: {best_params[0]}")
        print(f"batch_size: {best_params[1]}")
        print(f"learning rate: {best_params[2]}")
        print(f"Best validation_loss: {best_val_loss}")
        print(f"Test loss for this config: {best_test_loss}")

        best_config_foldername = f"num_units_{best_params[0]}_batch_size_{best_params[1]}_lr_{best_params[2]}"
        src_best_folder = os.path.join(rundir, best_config_foldername)
        dst_best_folder = os.path.join(rundir, "best_hparams")  

        if os.path.exists(dst_best_folder):
            shutil.rmtree(dst_best_folder)
        shutil.copytree(src_best_folder, dst_best_folder)

        with open(os.path.join(dst_best_folder, "best_summary.txt"), "w") as fsum:
            fsum.write("mejor configuración de hiperparámetros\n")
            fsum.write(f"num_units: {best_params[0]}\n")
            fsum.write(f"batch_size: {best_params[1]}\n")
            fsum.write(f"learning rate: {best_params[2]}\n")
            fsum.write(f"best validation_loss: {best_val_loss}\n")
            fsum.write(f"test loss (mejor config): {best_test_loss}\n")
    else:
        print("No valid configuration found during the sweep.")
