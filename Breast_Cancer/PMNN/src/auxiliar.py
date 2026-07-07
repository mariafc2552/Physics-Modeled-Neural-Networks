import torch
import os 
import matplotlib.pyplot as plt
from copy import deepcopy


def longtiming(secs):
    """
    Function for expressing long time intervals, from seconds into hours:minutes:seconds.

    Inputs:
        * secs: interval in seconds (as given by `time.time()`).

    Outputs:
        * Formatted string of hours : minutes : seconds.
    """
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours): 3d}:{int(minutes):02d}:{int(seconds):02d}"



class EarlyStopper:
    """
    General class for implementing early stopping in training.
    """
    def __init__(self, patience, min_delta):
        """
        Initializer for EarlyStopper.
        Inputs:
            * patience: number of calls (epochs) without improvement before sending stopping signal.
            * min_delta: minimum improvement to achieve before resetting step counter.
        """
        self.cnt = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = torch.inf
        self.update_loss = torch.inf
        self.best_model = None

    def __call__(self, loss, state_dict):
        """
        Check if early stopping must be applied.
        Inputs:
            * loss: validation loss or metric to minimize.
            * state_dict: training model's state to be stored.
        Outputs:
            * Boolean signal for stopping (True = 'stop').
        """
        if loss < self.best_loss:
            self.best_model = deepcopy(state_dict)
            self.best_loss = loss

        if loss < self.update_loss:
            self.update_loss = loss - self.min_delta
            self.cnt = 0
            return False
        if self.patience == 0:
            return False
        else:
            self.cnt += 1
            if self.cnt == self.patience:
                return True
            else:
                return False

    def reset(self):
        """
        Reset early stopper.
        """
        self.cnt = 0
        self.best_loss = torch.inf
        self.update_loss = torch.inf
        self.best_model = None

    def save(self, filename):
        """
        Save best model (state_dict) to file.
        Inputs:
            * filename: path to model file.
        """
        torch.save(self.best_model, filename)


def save_loss_curves(epoch_losses, val_losses, t_step, t_end, batch_size, lr, results_folder):
    """
    Generate and save a plot of training and validation loss curves.

    Inputs:
        * epoch_losses: list or array of training losses per epoch.
        * val_losses: list or array of validation losses per epoch.
        * t_step: value of the hyperparameter t_step.
        * t_end: value of the hyperparameter t_end.
        * batch_size: batch size used in training.
        * lr: learning rate of the optimizer.
        * results_folder: directory where the plot will be saved.

    Outputs:
        * Saves a `.png` file in the given folder, with a filename including
          hyperparameter values.
        * Returns the filename for traceability.
    """

    if not epoch_losses or not val_losses:
        raise ValueError("Loss lists cannot be empty.")

    if len(epoch_losses) != len(val_losses):
        raise ValueError(f"Length mismatch: {len(epoch_losses)} (train) vs {len(val_losses)} (valid).")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.plot(epoch_losses, label="Training Loss", color="tab:blue", linewidth=2.2, marker="o", markersize=4)
    plt.plot(val_losses, label="Validation Loss", color="tab:orange", linewidth=2.2, marker="s", markersize=4)

    plt.title(
        f"Training and Validation Loss\n"
        f"t_step={t_step}, t_end={t_end}, batch_size={batch_size}, lr={lr}",
        fontsize=14, fontweight="bold"
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11, loc="best", frameon=True, shadow=True)

    plt.tight_layout()

    filename = os.path.join(
        results_folder,
        f"t_step_{t_step}_t_end_{t_end}_batch_size_{batch_size}_lr_{lr}_loss_graph.png"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return filename
