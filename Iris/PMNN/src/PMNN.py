import logging
import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _unwrap_grid_value(value: Any) -> Any:
    """
    Utility function for configurations where a hyperparameter is stored
    as a one-element list due to grid-search formatting.

    Examples:
        [4]     -> 4
        [2]     -> 2
        [0.05]  -> 0.05
        4       -> 4
    """
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


class FitzHughNagumoSolver(nn.Module):
    """
    Discrete-time FitzHugh--Nagumo ODE solver.

    This module integrates the FitzHugh--Nagumo system using explicit Euler.

    The hidden state is two-dimensional:

        v(t): membrane-potential-like variable
        w(t): recovery variable

    The solver receives the initial conditions v0 and w0 generated from the
    input features and returns the terminal state [v_T, w_T].

    The implementation accepts float-valued dt and t_end to allow stable
    integration with small time steps.
    """

    def __init__(
        self,
        *,
        dt: float = 0.05,
        t_end: float = 1.0,
        a: float = 0.2,
        b: float = 0.02,
        g: float = 3.0,
        I: float = 0.0,
        state_clip: float = 20.0,
        device=None,
        dtype=None,
    ):
        """
        Initializer for FitzHughNagumoSolver.

        Args:
            * dt: integration time step.
            * t_end: total simulated internal time.
            * a: FHN parameter controlling the cubic nullcline.
            * b: FHN parameter scaling the recovery dynamics.
            * g: FHN coupling parameter between v and w.
            * I: constant input current.
            * state_clip: optional clipping value for numerical stability.
            * device: target device.
            * dtype: target floating-point dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        dt = float(_unwrap_grid_value(dt))
        t_end = float(_unwrap_grid_value(t_end))

        if dt <= 0.0:
            raise ValueError(f"`dt` must be positive, got {dt}.")

        if t_end <= 0.0:
            raise ValueError(f"`t_end` must be positive, got {t_end}.")

        if dt > t_end:
            raise ValueError(
                f"`dt` must be less than or equal to `t_end`, got dt={dt}, t_end={t_end}."
            )

        num_steps = int(math.ceil(t_end / dt))
        effective_dt = t_end / num_steps

        self.num_steps = num_steps
        self.state_clip = None if state_clip is None else float(state_clip)

        self.register_buffer("dt", torch.tensor(effective_dt, **factory_kwargs))
        self.register_buffer("t_end", torch.tensor(t_end, **factory_kwargs))
        self.register_buffer("a", torch.tensor(float(a), **factory_kwargs))
        self.register_buffer("b", torch.tensor(float(b), **factory_kwargs))
        self.register_buffer("g", torch.tensor(float(g), **factory_kwargs))
        self.register_buffer("I", torch.tensor(float(I), **factory_kwargs))

    def forward(self, v_init: torch.Tensor, w_init: torch.Tensor) -> torch.Tensor:
        """
        Forward Euler integration for the FitzHugh--Nagumo system.

        Args:
            * v_init: initial v state, shape (batch_size,)
            * w_init: initial w state, shape (batch_size,)

        Returns:
            * terminal_state: tensor [v_T, w_T], shape (batch_size, 2)
        """
        v = v_init.clone()
        w = w_init.clone()

        for _ in range(self.num_steps):
            dv_dt = self.I - v * (v - self.a) * (v - 1.0) - w
            dw_dt = self.b * (v - self.g * w)

            v = v + self.dt * dv_dt
            w = w + self.dt * dw_dt

            if self.state_clip is not None:
                v = torch.clamp(v, -self.state_clip, self.state_clip)
                w = torch.clamp(w, -self.state_clip, self.state_clip)

        terminal_state = torch.stack([v, w], dim=1)

        return terminal_state


class PMNNBlock(nn.Module):
    """
    FitzHugh--Nagumo-based PMNN for Iris classification.

    The Iris dataset has:

        input_size = 4
        output_size = 3

    The model expects standardized tabular features with shape:

        (batch_size, 4)

    Architecture:

        input features
            -> Linear(input_size, hidden_size)
            -> LayerNorm(hidden_size)
            -> activation
            -> bounded two-dimensional initial condition [v0, w0]
            -> FitzHugh--Nagumo solver
            -> Linear(2, output_size)

    The output is a tensor of raw logits with shape:

        (batch_size, 3)

    Therefore, the correct loss function is torch.nn.CrossEntropyLoss.
    """

    def __init__(
        self,
        *,
        input_size: int = 4,
        hidden_size: int = 2,
        output_size: int = 3,
        activation=nn.SiLU,
        dt: float = 0.05,
        t_end: float = 1.0,
        a: float = 0.2,
        b: float = 0.02,
        g: float = 3.0,
        I: float = 0.0,
        initial_state_scale: float = 1.0,
        state_clip: float = 20.0,
        use_layer_norm: bool = True,
        batch_size=None,
        device=None,
        dtype=None,
    ):
        """
        Initializer for PMNNBlock.

        Args:
            * input_size: number of input features. For Iris, use 4.
            * hidden_size: size of the pre-dynamical embedding. For a 2D FHN
              PMNN, use 2.
            * output_size: number of classes. For Iris, use 3.
            * activation: activation function after the hidden projection.
            * dt: Euler integration time step.
            * t_end: terminal internal integration time.
            * a, b, g, I: FitzHugh--Nagumo parameters.
            * initial_state_scale: scale applied to bounded initial conditions.
            * state_clip: clipping value inside the ODE solver.
            * use_layer_norm: whether to apply LayerNorm after the hidden linear layer.
            * batch_size: optional parameter accepted for YAML compatibility.
            * device: target device.
            * dtype: target floating-point dtype.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(_unwrap_grid_value(input_size))
        self.hidden_size = int(_unwrap_grid_value(hidden_size))
        self.output_size = int(_unwrap_grid_value(output_size))
        self.initial_state_scale = float(_unwrap_grid_value(initial_state_scale))
        self.use_layer_norm = bool(_unwrap_grid_value(use_layer_norm))

        if self.hidden_size < 2:
            raise ValueError(
                "hidden_size must be at least 2 because the FitzHugh--Nagumo "
                "system requires two initial conditions: v0 and w0."
            )

        if self.hidden_size != 2:
            logger.warning(
                "hidden_size=%s was provided, but this PMNN uses only the first "
                "two components as FHN initial conditions. For a strictly "
                "two-dimensional PMNN, use hidden_size=2.",
                self.hidden_size,
            )

        self.hidden = nn.Linear(
            self.input_size,
            self.hidden_size,
            bias=True,
            **factory_kwargs,
        )

        if self.use_layer_norm:
            self.hidden_norm = nn.LayerNorm(
                self.hidden_size,
                **factory_kwargs,
            )
        else:
            self.hidden_norm = nn.Identity()

        self.hidden_act = activation()

        self.solver = FitzHughNagumoSolver(
            dt=dt,
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
            * x: input tensor with shape (batch_size, 4)

        Returns:
            * logits: tensor with shape (batch_size, 3)
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

        h = self.hidden(x)
        h = self.hidden_norm(h)
        h = self.hidden_act(h)

        # Bound the initial state to improve numerical stability before the ODE.
        h = torch.tanh(h) * self.initial_state_scale

        v0 = h[:, 0]
        w0 = h[:, 1]

        terminal_state = self.solver(v0, w0)

        logits = self.output(terminal_state)

        return logits

    def save(self, file: str):
        """
        Save block state.

        Args:
            * file: path to state file.
        """
        torch.save(deepcopy(self.state_dict()), file)

    def load(self, file: str, map_location=None):
        """
        Load block state.

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

