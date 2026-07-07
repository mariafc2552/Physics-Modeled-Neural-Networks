import logging
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn
from torchdiffeq import odeint

logger = logging.getLogger(__name__)


def _unwrap_grid_value(value: Any) -> Any:
    """
    Utility function for configurations where a hyperparameter is stored
    as a one-element list due to grid-search formatting.

    Examples:
        [30]    -> 30
        [2]     -> 2
        [15]    -> 15
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]

    return value


def _resolve_hidden_dim(hidden_dim: Optional[Any], hidden_dims: Optional[Any]) -> int:
    """
    Resolve hidden dimension from either hidden_dim or hidden_dims.

    This is useful because some configuration files use:

        hidden_dims: [5, 10, 15, 20]

    while the model itself receives one selected value at a time.
    """
    if hidden_dim is None and hidden_dims is None:
        return 15

    if hidden_dim is None:
        hidden_dim = hidden_dims

    hidden_dim = _unwrap_grid_value(hidden_dim)

    if isinstance(hidden_dim, (list, tuple)):
        if len(hidden_dim) == 0:
            raise ValueError("hidden_dim/hidden_dims cannot be empty.")

        if len(hidden_dim) > 1:
            raise ValueError(
                "NODEblock received several hidden dimensions at once. "
                "The main.py file should iterate over config['model']['hidden_dims'] "
                "and pass one value at a time."
            )

        hidden_dim = hidden_dim[0]

    hidden_dim = int(hidden_dim)

    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

    return hidden_dim


class ODEfunc(nn.Module):
    """
    Time-dependent neural dynamics f(t, x) for a Neural ODE.

    The state x has shape:

        (batch_size, hidden_dim)

    Time t is concatenated to the state, so the internal MLP receives:

        (batch_size, hidden_dim + 1)
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: nn.Module = nn.Tanh,
        use_layer_norm: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Initializer for ODEfunc.

        Args:
            * dim: dimensionality of the ODE state.
            * hidden_dim: number of hidden units inside the ODE function.
            * num_layers: number of linear layers in the ODE function.
              Values from your config are [2, 3].
            * activation: activation function used inside the ODE function.
            * use_layer_norm: whether to normalize [x, t] before the MLP.
            * device: target device.
            * dtype: target floating-point dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.dim = int(_unwrap_grid_value(dim))
        self.hidden_dim = int(_unwrap_grid_value(hidden_dim))
        self.num_layers = int(_unwrap_grid_value(num_layers))
        self.use_layer_norm = bool(_unwrap_grid_value(use_layer_norm))

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}.")

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")

        if self.num_layers < 2:
            raise ValueError(
                "num_layers must be at least 2. "
                f"Received num_layers={self.num_layers}."
            )

        input_dim = self.dim + 1

        layers = []

        for layer_index in range(self.num_layers - 1):
            in_features = input_dim if layer_index == 0 else self.hidden_dim
            out_features = self.hidden_dim

            layers.append(
                nn.Linear(
                    in_features,
                    out_features,
                    **factory_kwargs,
                )
            )
            layers.append(activation())

        layers.append(
            nn.Linear(
                self.hidden_dim,
                self.dim,
                **factory_kwargs,
            )
        )

        self.net = nn.Sequential(*layers)

        if self.use_layer_norm:
            self.norm = nn.LayerNorm(
                input_dim,
                **factory_kwargs,
            )
        else:
            self.norm = nn.Identity()

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODE dynamics.

        Args:
            * t: scalar tensor representing time.
            * x: tensor with shape (batch_size, dim).

        Returns:
            * dx/dt: tensor with shape (batch_size, dim).
        """
        if x.dim() != 2:
            raise ValueError(
                "ODEfunc expects x with shape (batch_size, dim), "
                f"but received {tuple(x.shape)}."
            )

        batch_size = x.shape[0]

        t_tensor = torch.ones(
            batch_size,
            1,
            device=x.device,
            dtype=x.dtype,
        ) * t

        xt = torch.cat([x, t_tensor], dim=1)
        xt = self.norm(xt)

        dxdt = self.net(xt)

        return dxdt


class ODEBlock(nn.Module):
    """
    Integrates the ODE defined by ODEfunc over a fixed interval [t0, t1].
    """

    def __init__(
        self,
        odefunc: ODEfunc,
        *,
        t0: float = 0.0,
        t1: float = 1.0,
        method: Optional[str] = None,
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        """
        Initializer for ODEBlock.

        Args:
            * odefunc: instance of ODEfunc.
            * t0: initial integration time.
            * t1: final integration time.
            * method: optional torchdiffeq solver method.
            * rtol: relative tolerance.
            * atol: absolute tolerance.
        """
        super().__init__()

        self.odefunc = odefunc
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)

        if self.t1 <= self.t0:
            raise ValueError(
                f"t1 must be greater than t0. Received t0={self.t0}, t1={self.t1}."
            )

        self.register_buffer(
            "integration_time",
            torch.tensor([self.t0, self.t1], dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ODEBlock.

        Args:
            * x: tensor with shape (batch_size, dim).

        Returns:
            * tensor with shape (batch_size, dim).
        """
        integration_time = self.integration_time.to(
            device=x.device,
            dtype=x.dtype,
        )

        out = odeint(
            self.odefunc,
            x,
            integration_time,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

        return out[-1]


class NODEblock(nn.Module):
    """
    Neural ODE model for Breast Cancer Wisconsin binary classification.

    Expected input:

        x.shape = (batch_size, 30)

    Default architecture:

        Linear(30, hidden_dim)
        ODEBlock(hidden_dim)
        Linear(hidden_dim, 2)

    Output:

        logits.shape = (batch_size, 2)

    Therefore, the recommended loss function is:

        torch.nn.CrossEntropyLoss()

    Do not apply sigmoid or softmax inside this model.
    """

    def __init__(
        self,
        input_size: int = 30,
        output_size: int = 2,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        *,
        hidden_dims: Optional[Any] = None,
        input_channels: Optional[int] = None,
        activation: nn.Module = nn.Tanh,
        use_layer_norm: bool = True,
        t0: float = 0.0,
        t1: float = 1.0,
        method: Optional[str] = None,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        batch_size=None,
        device=None,
        dtype=None,
    ):
        """
        Initializer for NODEblock.

        Args:
            * input_size: number of tabular input features.
              For Breast Cancer, use 30.
            * output_size: number of output classes.
              For CrossEntropyLoss, use 2.
            * hidden_dim: selected hidden dimension for this run.
            * num_layers: number of layers in the ODE function.
            * hidden_dims: optional alias for hidden_dim.
            * input_channels: accepted for compatibility with older image configs.
              It is not used for tabular Breast Cancer data.
            * activation: activation function inside the ODE function.
            * use_layer_norm: whether to use LayerNorm inside ODEfunc.
            * t0: initial integration time.
            * t1: final integration time.
            * method: optional torchdiffeq solver method.
            * rtol: relative tolerance.
            * atol: absolute tolerance.
            * batch_size: optional argument accepted for YAML compatibility.
            * device: target device.
            * dtype: target floating-point dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(_unwrap_grid_value(input_size))
        self.output_size = int(_unwrap_grid_value(output_size))
        self.hidden_dim = _resolve_hidden_dim(hidden_dim, hidden_dims)
        self.num_layers = int(_unwrap_grid_value(num_layers))

        self.input_channels = input_channels
        self.batch_size = batch_size

        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}.")

        if self.output_size <= 1:
            raise ValueError(
                "output_size must be greater than 1 when using CrossEntropyLoss. "
                f"Received output_size={self.output_size}."
            )

        if self.num_layers < 2:
            raise ValueError(
                "num_layers must be at least 2. "
                f"Received num_layers={self.num_layers}."
            )

        self.input_layer = nn.Linear(
            self.input_size,
            self.hidden_dim,
            **factory_kwargs,
        )

        self.input_activation = nn.Tanh()

        self.odeblock = ODEBlock(
            ODEfunc(
                dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                activation=activation,
                use_layer_norm=use_layer_norm,
                device=device,
                dtype=dtype,
            ),
            t0=t0,
            t1=t1,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        self.output_layer = nn.Linear(
            self.hidden_dim,
            self.output_size,
            **factory_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for NODEblock.

        Args:
            * x: input tensor with shape (batch_size, 30).

        Returns:
            * logits: tensor with shape (batch_size, 2).
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        if x.dim() != 2:
            raise ValueError(
                "Expected input tensor with shape (batch_size, input_size), "
                f"but received {tuple(x.shape)}."
            )

        if x.size(1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"but received {x.size(1)}."
            )

        h0 = self.input_layer(x)
        h0 = self.input_activation(h0)

        h1 = self.odeblock(h0)

        logits = self.output_layer(h1)

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


class NODEBlock(NODEblock):
    """
    Alias kept for compatibility with alternative import styles.
    """
    pass

