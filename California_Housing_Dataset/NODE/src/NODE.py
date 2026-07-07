import logging
import torch
import torch.nn as nn
from torchdiffeq import odeint

logger = logging.getLogger(__name__)


class ODEfunc(nn.Module):
    """
    Defines the dynamics f(t, x) of the Neural ODE using a time-dependent multi-layer perceptron.
    """

    def __init__(self, dim, hidden_dim, num_layers):
        """
        Initializer for `ODEfunc`.

        Args:
            * dim: dimensionality of the input and output vectors.
            * hidden_dim: number of units in the hidden layers.
            * num_layers: number of MLP layers (>= 2).
        """
        super(ODEfunc, self).__init__()
        layers = []
        in_dim = dim + 1  # +1 for time input
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, dim))

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, t, x):
        """
        Forward pass of the ODE dynamics.

        Args:
            * t: scalar tensor representing time.
            * x: input tensor of shape (batch_size, dim).

        Returns:
            * output tensor of shape (batch_size, dim).
        """
        batch_size = x.shape[0]
        t_tensor = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype) * t
        xt = torch.cat([x, t_tensor], dim=1)
        out = self.norm(xt)
        out = self.net(out)
        return out


class ODEBlock(nn.Module):
    """
    Integrates the ODE defined by `ODEfunc` over a fixed interval [0, 1].
    """

    def __init__(self, odefunc):
        """
        Initializer for `ODEBlock`.

        Args:
            * odefunc: instance of ODEfunc.
        """
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        """
        Forward pass through the ODEBlock.

        Args:
            * x: input tensor of shape (batch_size, dim).

        Returns:
            * output tensor of shape (batch_size, dim).
        """
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time)
        return out[1]


class NODEblock(nn.Module):
    """
    Block implementing a NODE model composed of a time-dependent MLP-based ODE and a final linear readout.
    """

    def __init__(self, input_size: int, output_size: int, hidden_dim: int, num_layers: int):
        """
        Initializer for `NODEblock`.

        Args:
            * input_size: dimensionality of the input vector.
            * hidden_size: dimensionality of the hidden dynamics and output.
        """
        super(NODEblock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size, hidden_dim)
        self.odeblock = ODEBlock(ODEfunc(dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers))
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for `NODEblock`.

        Args:
            * x: input tensor of shape (batch_size, input_size).

        Returns:
            * y_hat: output tensor of shape (batch_size, 1).
        """
        h0 = self.input_layer(x)
        h1 = self.odeblock(h0)
        y_hat = self.output_layer(h1)
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
