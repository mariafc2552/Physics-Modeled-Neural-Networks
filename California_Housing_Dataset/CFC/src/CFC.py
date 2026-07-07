import logging
import torch
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import CfC

logger = logging.getLogger(__name__)


class CFCblock(nn.Module):
    """
    Block implementing a Closed-form Continuous-time (CfC) model using AutoNCP wiring.

    """

    def __init__(self, input_size: int, num_units: int, output_size: int):
        """
        Initializer for `CFCblock`.

        Args:
            * input_size: dimensionality of the input vector.
            * num_units: number of internal units used for AutoNCP topology.
            * output_size: dimensionality of the output vector.
        """
        super(CFCblock, self).__init__()

        self.input_size = input_size
        self.hidden_size = num_units
        self.output_size = output_size

        #wiring = AutoNCP(num_units, output_size)

        self.cfc_model = CfC(input_size, num_units, batch_first=True)
        self.fc_out = nn.Linear(num_units, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for `CFCblock`.

        Args:
            * x: input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            * y_hat: output tensor of shape (batch_size, sequence_length, output_size).
        """
        x = x.unsqueeze(1)
        y_hat, _ = self.cfc_model(x)
        y_hat = y_hat[:, -1, :]
        y_hat = self.fc_out(y_hat)
        return y_hat

    def save(self, file: str):
        """
        Save the model state to file.

        Args:
            * file: path to the destination file.
        """
        torch.save(self.state_dict(), file)

    def load(self, file: str):
        """
        Load the model state from file.

        Args:
            * file: path to the source file.
        """
        self.load_state_dict(torch.load(file, weights_only=True))
