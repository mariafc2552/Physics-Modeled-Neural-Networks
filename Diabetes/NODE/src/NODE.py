import logging
from copy import deepcopy
from typing import Any, Optional, Union, Type

import torch
import torch.nn as nn
from torchdiffeq import odeint

logger = logging.getLogger(__name__)


def _unwrap_grid_value(value: Any) -> Any:
    """
    Extract scalar values from one-element lists.

    Examples:
        [10] -> 10
        [1]  -> 1
        [15] -> 15
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]

    return value


def _resolve_hidden_dim(hidden_dim=None, hidden_dims=None, default: int = 15) -> int:
    """
    Resolve hidden_dim from either hidden_dim or hidden_dims.

    In the YAML configuration, the grid-search parameter is usually:

        hidden_dims: [5, 10, 15, 20]

    The main.py should iterate over this list and pass a single value as
    hidden_dim. This function also allows hidden_dims to be passed directly
    if it contains only one value.
    """
    if hidden_dim is not None:
        return int(_unwrap_grid_value(hidden_dim))

    if hidden_dims is None:
        return int(default)

    hidden_dims = _unwrap_grid_value(hidden_dims)

    if isinstance(hidden_dims, int):
        return int(hidden_dims)

    if isinstance(hidden_dims, (list, tuple)):
        if len(hidden_dims) == 1:
            return int(hidden_dims[0])

        raise ValueError(
            "hidden_dims contains multiple values. The main.py should iterate "
            "over config['model']['hidden_dims'] and pass one value as hidden_dim."
        )

    raise TypeError("hidden_dim or hidden_dims must be an int or a list of ints.")


def _as_bool(value: Any) -> bool:
    """
    Convert YAML-compatible values to bool.
    """
    value = _unwrap_grid_value(value)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "y"}

    return bool(value)


def _get_activation(
    activation: Union[str, Type[nn.Module], nn.Module],
) -> nn.Module:
    """
    Return activation module from string, class or instance.
    """
    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    if isinstance(activation, str):
        name = activation.lower()

        if name == "tanh":
            return nn.Tanh()

        if name == "relu":
            return nn.ReLU()

        if name == "silu":
            return nn.SiLU()

        if name == "gelu":
            return nn.GELU()

        if name == "elu":
            return nn.ELU()

        raise ValueError(
            f"Unknown activation '{activation}'. "
            "Available options: tanh, relu, silu, gelu, elu."
        )

    raise TypeError(
        "activation must be a string, a torch.nn.Module class, "
        "or a torch.nn.Module instance."
    )


class ODEfunc(nn.Module):
    """
    Time-dependent ODE dynamics for a Neural ODE.

    The dynamics are represented by a multilayer perceptron:

        dx/dt = f(t, x)

    Since the function is explicitly time-dependent, the scalar time t is
    concatenated to the hidden state x before being passed through the MLP.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: Union[str, Type[nn.Module], nn.Module] = "tanh",
        use_layer_norm: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Initializer for ODEfunc.

        Args:
            * dim: dimension of the hidden state.
            * hidden_dim: width of the internal MLP.
            * num_layers: number of linear layers in the ODE function.
            * activation: activation function used between linear layers.
            * use_layer_norm: whether to normalize the concatenated [x, t].
            * device: target device.
            * dtype: target tensor dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.dim = int(_unwrap_grid_value(dim))
        self.hidden_dim = int(_unwrap_grid_value(hidden_dim))
        self.num_layers = int(_unwrap_grid_value(num_layers))
        self.use_layer_norm = _as_bool(use_layer_norm)

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}.")

        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}.")

        if self.num_layers < 2:
            raise ValueError(
                f"num_layers must be >= 2, got {self.num_layers}."
            )

        input_dim = self.dim + 1

        layers = []
        previous_dim = input_dim

        for _ in range(self.num_layers - 1):
            layers.append(
                nn.Linear(
                    previous_dim,
                    self.hidden_dim,
                    **factory_kwargs,
                )
            )
            layers.append(_get_activation(activation))
            previous_dim = self.hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                self.dim,
                **factory_kwargs,
            )
        )

        self.net = nn.Sequential(*layers)

        if self.use_layer_norm:
            self.norm = nn.LayerNorm(input_dim, **factory_kwargs)
        else:
            self.norm = nn.Identity()

    def forward(self, t, x):
        """
        Forward pass of the ODE dynamics.

        Args:
            * t: scalar tensor representing time.
            * x: hidden state tensor with shape (batch_size, dim).

        Returns:
            * dxdt: tensor with shape (batch_size, dim).
        """
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
    Integrates the ODE defined by ODEfunc over a fixed time interval.
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
            * odefunc: ODEfunc instance.
            * t0: initial integration time.
            * t1: final integration time.
            * method: optional torchdiffeq integration method.
            * rtol: relative tolerance.
            * atol: absolute tolerance.
        """
        super().__init__()

        self.odefunc = odefunc
        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)

        self.register_buffer(
            "integration_time",
            torch.tensor([float(t0), float(t1)], dtype=torch.float32),
        )

    def forward(self, x):
        """
        Forward pass through the ODEBlock.

        Args:
            * x: tensor with shape (batch_size, dim).

        Returns:
            * out: tensor with shape (batch_size, dim).
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
    Neural ODE model for the Diabetes regression dataset.

    Data interpretation:

        x.shape = (batch_size, 10)

    Architecture:

        x
        -> Linear(input_size, hidden_dim)
        -> activation
        -> ODEBlock
        -> Linear(hidden_dim, output_size)

    For Diabetes regression:

        input_size = 10
        output_size = 1

    The model returns a continuous scalar prediction. The associated main.py
    should use a regression loss such as:

        nn.MSELoss()
    """

    def __init__(
        self,
        input_size: int = 10,
        output_size: int = 1,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        *,
        hidden_dims: Optional[Any] = None,
        input_channels: Optional[int] = None,
        activation: Union[str, Type[nn.Module], nn.Module] = "tanh",
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
            * input_size: number of input features.
              For Diabetes, use 10.
            * output_size: number of output variables.
              For Diabetes regression, use 1.
            * hidden_dim: hidden dimension used by the NODE.
            * num_layers: number of linear layers inside the ODE function.
            * hidden_dims: optional alias for hidden_dim, useful for YAML compatibility.
            * input_channels: accepted for compatibility with previous image-based
              configurations. It is ignored for tabular data.
            * activation: activation function.
            * use_layer_norm: whether to use LayerNorm in the ODE function.
            * t0: initial integration time.
            * t1: final integration time.
            * method: optional ODE solver method.
            * rtol: relative tolerance.
            * atol: absolute tolerance.
            * batch_size: accepted for YAML compatibility, not used internally.
            * device: target device.
            * dtype: target tensor dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(_unwrap_grid_value(input_size))
        self.output_size = int(_unwrap_grid_value(output_size))
        self.hidden_dim = _resolve_hidden_dim(hidden_dim, hidden_dims, default=15)
        self.num_layers = int(_unwrap_grid_value(num_layers))
        self.input_channels = input_channels
        self.batch_size = batch_size

        if self.input_size <= 0:
            raise ValueError(
                f"input_size must be positive, got {self.input_size}."
            )

        if self.output_size <= 0:
            raise ValueError(
                f"output_size must be positive, got {self.output_size}."
            )

        if self.output_size != 1:
            raise ValueError(
                "For the Diabetes regression dataset, output_size must be 1. "
                f"Received output_size={self.output_size}. "
                "Change your config.yaml from output_size: 2 to output_size: 1."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {self.hidden_dim}."
            )

        if self.num_layers < 2:
            raise ValueError(
                f"num_layers must be >= 2, got {self.num_layers}."
            )

        self.input_layer = nn.Linear(
            self.input_size,
            self.hidden_dim,
            **factory_kwargs,
        )

        self.input_activation = _get_activation(activation)

        self.odeblock = ODEBlock(
            ODEfunc(
                dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                activation=activation,
                use_layer_norm=use_layer_norm,
                **factory_kwargs,
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
            * x: input tensor with shape (batch_size, 10).

        Returns:
            * y_hat: output tensor with shape (batch_size, 1).
        """
        if x.dim() == 1:
            x = x.view(1, -1)

        elif x.dim() > 2:
            x = x.view(x.size(0), -1)

        if x.size(1) != self.input_size:
            raise ValueError(
                f"Expected input with {self.input_size} features, "
                f"but received tensor with shape {tuple(x.shape)}."
            )

        h0 = self.input_layer(x)
        h0 = self.input_activation(h0)

        h1 = self.odeblock(h0)

        y_hat = self.output_layer(h1)

        return y_hat

    def save(self, file: str):
        """
        Save model state.

        Args:
            * file: path to destination file.
        """
        torch.save(deepcopy(self.state_dict()), file)

    def load(self, file: str, map_location=None):
        """
        Load model state.

        Args:
            * file: path to source file.
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

