import logging
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
from ncps.torch import CfC

logger = logging.getLogger(__name__)


def _unwrap_grid_value(value: Any) -> Any:
    """
    Utility function for configurations where a hyperparameter is stored
    as a one-element list due to grid-search formatting.

    Examples:
        [4]      -> 4
        [64]     -> 64
        [2]      -> 2
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]

    return value


class CFCblock(nn.Module):
    """
    Closed-form Continuous-time Neural Network for Banknote Authentication
    binary classification.

    Data interpretation:

        Original tabular input:
            x.shape = (batch_size, 4)

        CfC input:
            x.shape = (batch_size, 1, 4)

    Therefore, each sample is interpreted as a sequence with one artificial
    time step and 4 input features.

    Architecture:

        input vector of 4 features
            -> unsqueeze temporal dimension
            -> CfC(input_size=4, units=num_units)
            -> last hidden representation
            -> Linear(num_units, output_size)

    For Banknote Authentication classification with CrossEntropyLoss:

        output_size = 2

    The model returns raw logits. Do not apply sigmoid or softmax inside
    this model when using torch.nn.CrossEntropyLoss.
    """

    def __init__(
        self,
        input_size: int = 4,
        num_units: int = 64,
        output_size: int = 2,
        *,
        batch_first: bool = True,
        batch_size=None,
        device=None,
        dtype=None,
    ):
        """
        Initializer for CFCblock.

        Args:
            * input_size: number of input features.
              For Banknote Authentication, use 4.
            * num_units: number of CfC hidden units.
              In your grid search: 64, 128, or 256.
            * output_size: number of output classes.
              For Banknote Authentication with CrossEntropyLoss, use 2.
            * batch_first: whether input shape is (batch, seq, features).
            * batch_size: optional argument accepted for YAML compatibility.
            * device: target device.
            * dtype: target floating-point dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(_unwrap_grid_value(input_size))
        self.num_units = int(_unwrap_grid_value(num_units))
        self.hidden_size = self.num_units
        self.output_size = int(_unwrap_grid_value(output_size))
        self.batch_first = bool(_unwrap_grid_value(batch_first))
        self.batch_size = batch_size

        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}.")

        if self.num_units <= 0:
            raise ValueError(f"num_units must be positive, got {self.num_units}.")

        if self.output_size <= 1:
            raise ValueError(
                "output_size must be greater than 1 when using CrossEntropyLoss. "
                f"Received output_size={self.output_size}."
            )

        if not self.batch_first:
            raise ValueError(
                "This implementation expects batch_first=True, i.e. "
                "input shape (batch_size, sequence_length, input_size)."
            )

        self.cfc_model = CfC(
            self.input_size,
            self.num_units,
            batch_first=True,
        )

        self.fc_out = nn.Linear(
            self.num_units,
            self.output_size,
            **factory_kwargs,
        )

    def _prepare_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare Banknote Authentication tabular data for CfC.

        Accepted inputs:

            1. x.shape = (batch_size, 4)
               This is converted to (batch_size, 1, 4).

            2. x.shape = (batch_size, 1, 4)
               This is already a valid CfC input.

        Returns:
            Tensor with shape (batch_size, 1, 4).
        """
        if x.dim() == 1:
            x = x.view(1, -1)

        if x.dim() == 2:
            if x.size(1) != self.input_size:
                raise ValueError(
                    f"Expected {self.input_size} input features, "
                    f"but received {x.size(1)}."
                )

            x = x.unsqueeze(1)

        elif x.dim() == 3:
            if x.size(-1) != self.input_size:
                raise ValueError(
                    f"Expected last dimension equal to input_size={self.input_size}, "
                    f"but received shape {tuple(x.shape)}."
                )

        else:
            x = x.view(x.size(0), -1)

            if x.size(1) != self.input_size:
                raise ValueError(
                    f"Expected {self.input_size} input features after flattening, "
                    f"but received {x.size(1)}."
                )

            x = x.unsqueeze(1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for CFCblock.

        Args:
            * x: input tensor with shape (batch_size, 4).

        Returns:
            * logits: tensor with shape (batch_size, 2).
        """
        x = self._prepare_sequence(x)

        y_hat, _ = self.cfc_model(x)

        if y_hat.dim() == 3:
            y_hat = y_hat[:, -1, :]

        logits = self.fc_out(y_hat)

        return logits

    def save(self, file: str):
        """
        Save the model state to file.

        Args:
            * file: path to the destination file.
        """
        torch.save(deepcopy(self.state_dict()), file)

    def load(self, file: str, map_location=None):
        """
        Load the model state from file.

        Args:
            * file: path to the source file.
            * map_location: optional torch map_location argument.
        """
        state_dict = torch.load(
            file,
            map_location=map_location,
            weights_only=True,
        )
        self.load_state_dict(state_dict)


class CfCBlock(CFCblock):
    """
    Alias kept for compatibility with alternative import styles.
    """
    pass
