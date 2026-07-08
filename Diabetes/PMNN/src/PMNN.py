import logging
from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _unwrap_grid_value(value: Any) -> Any:
    """
    Extract scalar values from one-element lists.

    This is useful when the YAML configuration stores some values as lists
    for grid-search compatibility.

    Examples:
        [10] -> 10
        [2]  -> 2
        [1]  -> 1
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]

    return value


def _get_activation(activation):
    """
    Return an activation module from either a class, an instance or a string.
    """
    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    if isinstance(activation, str):
        name = activation.lower()

        if name == "relu":
            return nn.ReLU()

        if name == "silu":
            return nn.SiLU()

        if name == "tanh":
            return nn.Tanh()

        if name == "gelu":
            return nn.GELU()

        if name == "elu":
            return nn.ELU()

        raise ValueError(
            f"Unknown activation '{activation}'. "
            "Available options: relu, silu, tanh, gelu, elu."
        )

    raise TypeError(
        "activation must be a torch.nn.Module instance, "
        "a torch.nn.Module class, or a string."
    )


class FitzHughNagumoSolver(nn.Module):
    """
    Discrete-time FitzHugh-Nagumo solver.

    This module integrates the FHN system using an explicit Euler scheme.

    The dynamics are:

        dv/dt = I - v(v - a)(v - 1) - w
        dw/dt = b(v - g w)

    For stability, this implementation uses `num_steps` Euler steps over
    the interval [0, t_end], with:

        dt = t_end / num_steps

    This is safer than using a very large Euler step directly.
    """

    def __init__(
        self,
        *,
        num_steps: int = 10,
        t_end: float = 10.0,
        a: float = 0.2,
        b: float = 0.02,
        g: float = 3.0,
        I: float = 0.0,
        state_clip: Optional[float] = 20.0,
        device=None,
        dtype=None,
    ):
        """
        Initializer for FitzHughNagumoSolver.

        Args:
            * num_steps: number of Euler integration steps.
            * t_end: final simulation time.
            * a, b, g, I: FitzHugh-Nagumo parameters.
            * state_clip: optional clipping value for v and w to avoid numerical
              explosions during large hyperparameter sweeps.
            * device: target device.
            * dtype: target floating-point dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        num_steps = int(_unwrap_grid_value(num_steps))
        t_end = float(_unwrap_grid_value(t_end))

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")

        if t_end <= 0.0:
            raise ValueError(f"t_end must be positive, got {t_end}.")

        self.num_steps = num_steps
        self.state_clip = state_clip

        self.register_buffer("t_end", torch.tensor(t_end, **factory_kwargs))
        self.register_buffer("dt", torch.tensor(t_end / num_steps, **factory_kwargs))

        self.register_buffer("a", torch.tensor(float(a), **factory_kwargs))
        self.register_buffer("b", torch.tensor(float(b), **factory_kwargs))
        self.register_buffer("g", torch.tensor(float(g), **factory_kwargs))
        self.register_buffer("I", torch.tensor(float(I), **factory_kwargs))

    def forward(self, v_init: torch.Tensor, w_init: torch.Tensor) -> torch.Tensor:
        """
        Forward Euler integration for the FHN system.

        Args:
            * v_init: initial membrane potential, shape (batch_size,)
            * w_init: initial recovery variable, shape (batch_size,)

        Returns:
            * terminal_state: tensor with shape (batch_size, 2)
              containing [v_T, w_T].
        """
        v = v_init.clone()
        w = w_init.clone()

        for _ in range(self.num_steps):
            dv_dt = self.I - v * (v - self.a) * (v - 1.0) - w
            dw_dt = self.b * (v - self.g * w)

            v = v + self.dt * dv_dt
            w = w + self.dt * dw_dt

            if self.state_clip is not None:
                clip_value = float(self.state_clip)
                v = torch.clamp(v, min=-clip_value, max=clip_value)
                w = torch.clamp(w, min=-clip_value, max=clip_value)

        terminal_state = torch.stack([v, w], dim=1)

        return terminal_state


class PMNNBlock(nn.Module):
    """
    Physics-inspired neural network block for the Diabetes regression dataset.

    Data interpretation:

        x.shape = (batch_size, 10)

    The model computes:

        x
        -> Linear(input_size, hidden_size)
        -> optional LayerNorm
        -> activation
        -> first two hidden components define FHN initial conditions:
              v0 = h[:, 0]
              w0 = h[:, 1]
        -> FitzHugh-Nagumo solver
        -> Linear(2, output_size)

    For the Diabetes dataset:

        input_size = 10
        output_size = 1

    The output is a continuous value, so the associated main.py should use:

        nn.MSELoss()

    or another regression loss.
    """

    def __init__(
        self,
        *,
        input_size: int = 10,
        hidden_size: int = 2,
        output_size: int = 1,
        activation=nn.SiLU,
        num_steps: Optional[int] = None,
        t_step: Optional[int] = None,
        dt: Optional[int] = None,
        t_end: float = 10.0,
        a: float = 0.2,
        b: float = 0.02,
        g: float = 3.0,
        I: float = 0.0,
        initial_state_scale: float = 1.0,
        state_clip: Optional[float] = 20.0,
        use_layer_norm: bool = True,
        batch_size=None,
        device=None,
        dtype=None,
    ):
        """
        Initializer for PMNNBlock.

        Args:
            * input_size: number of input variables.
              For Diabetes, use 10.
            * hidden_size: size of the pre-dynamical hidden embedding.
              Must be at least 2 because h[:, 0] and h[:, 1] are used as
              FHN initial conditions.
            * output_size: number of output variables.
              For Diabetes regression, use 1.
            * activation: activation function before the dynamical system.
            * num_steps: number of Euler integration steps.
            * t_step: alias for num_steps, useful for YAML compatibility.
            * dt: legacy alias. If provided and num_steps/t_step are not given,
              it is interpreted as the number of Euler steps for compatibility
              with previous PMNN configurations.
            * t_end: final integration time.
            * a, b, g, I: FitzHugh-Nagumo parameters.
            * initial_state_scale: multiplicative factor applied to v0 and w0.
            * state_clip: optional clipping value for numerical stability.
            * use_layer_norm: whether to use LayerNorm after the hidden layer.
            * batch_size: accepted for YAML compatibility, not used internally.
            * device: target device.
            * dtype: target floating-point dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(_unwrap_grid_value(input_size))
        self.hidden_size = int(_unwrap_grid_value(hidden_size))
        self.output_size = int(_unwrap_grid_value(output_size))
        self.initial_state_scale = float(_unwrap_grid_value(initial_state_scale))
        self.batch_size = batch_size

        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}.")

        if self.hidden_size < 2:
            raise ValueError(
                "hidden_size must be at least 2 because the first two hidden "
                "components are used as initial conditions v0 and w0."
            )

        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {self.output_size}.")

        if num_steps is None:
            if t_step is not None:
                num_steps = t_step
            elif dt is not None:
                num_steps = dt
            else:
                num_steps = 10

        num_steps = int(_unwrap_grid_value(num_steps))
        t_end = float(_unwrap_grid_value(t_end))

        self.hidden = nn.Linear(
            self.input_size,
            self.hidden_size,
            bias=True,
            **factory_kwargs,
        )

        if use_layer_norm:
            self.hidden_norm = nn.LayerNorm(self.hidden_size, **factory_kwargs)
        else:
            self.hidden_norm = nn.Identity()

        self.hidden_act = _get_activation(activation)

        self.solver = FitzHughNagumoSolver(
            num_steps=num_steps,
            t_end=t_end,
            a=a,
            b=b,
            g=g,
            I=I,
            state_clip=state_clip,
            **factory_kwargs,
        )

        self.output = nn.Linear(
            2,
            self.output_size,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for PMNNBlock.

        Args:
            * x: input tensor with shape (batch_size, 10)

        Returns:
            * y_hat: prediction tensor with shape (batch_size, 1)
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

        h = self.hidden(x)
        h = self.hidden_norm(h)
        h = self.hidden_act(h)

        v0 = self.initial_state_scale * h[:, 0]
        w0 = self.initial_state_scale * h[:, 1]

        terminal_state = self.solver(v0, w0)

        y_hat = self.output(terminal_state)

        return y_hat

    def save(self, file: str):
        """
        Save model state.

        Args:
            * file: path to state file.
        """
        torch.save(deepcopy(self.state_dict()), file)

    def load(self, file: str, map_location=None):
        """
        Load model state.

        Args:
            * file: path to state file.
            * map_location: optional torch map_location argument.
        """
        state_dict = torch.load(
            file,
            map_location=map_location,
            weights_only=True,
        )

        self.load_state_dict(state_dict)


class PMNNblock(PMNNBlock):
    """
    Alias kept for compatibility with alternative import styles.
    """
    pass

