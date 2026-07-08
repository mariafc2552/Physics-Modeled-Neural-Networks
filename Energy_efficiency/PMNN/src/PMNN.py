import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FitzHughNagumoSolver(nn.Module):
    """
    Discrete-time FitzHugh-Nagumo (FHN) ODE solver module.

    This module integrates the FHN system using explicit Euler with a fixed
    step size `dt` for a total simulated duration `t_end`.

    """

    def __init__(
        self,
        *,
        dt: int = 5,
        t_end: int = 10,
        a: float = 0.2,
        b: float = 0.02,
        g: float = 3.0,
        I: float = 0.0,
        device=None,
        dtype=None,
    ):
        """
        Initializer for `FitzHughNagumoSolver`.

        Inputs:
            * dt: integration time step.
            * t_end: total simulated time.
            * a: FHN parameter controlling the cubic nullcline.
            * b: FHN parameter scaling the recovery variable dynamics.
            * g: FHN coupling between `v` and `w`.
            * I: input current (constant drive).
            * device
            * dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        assert isinstance(dt, int) and dt > 0, f"`dt` must be a positive integer, got {dt}."
        assert isinstance(t_end, int) and t_end > 0, f"`t_end` must be a positive integer, got {t_end}."

        self.register_buffer("dt", torch.tensor(dt, **factory_kwargs))
        self.register_buffer("t_end", torch.tensor(t_end, **factory_kwargs))
        self.register_buffer("a", torch.tensor(float(a), **factory_kwargs))
        self.register_buffer("b", torch.tensor(float(b), **factory_kwargs))
        self.register_buffer("g", torch.tensor(float(g), **factory_kwargs))
        self.register_buffer("I", torch.tensor(float(I), **factory_kwargs))

        self.t_points = int(self.t_end.item() // self.dt.item())

    def forward(self, v_init: torch.Tensor, w_init: torch.Tensor) -> torch.Tensor:
        """
        Forward Euler integration for the FHN system.

        Inputs:
            * v_init: initial membrane potential, shape (N,) or (batch,).
            * w_init: initial recovery variable, shape (N,) or (batch,).

        Outputs:
            * Final state tensor with stacked components [v, w], shape (N, 2).
        """
        v = v_init.clone()
        w = w_init.clone()

        for _ in range(self.t_points):
            dv_dt = self.I - v * (v - self.a) * (v - 1.0) - w
            dw_dt = self.b * (v - self.g * w)
            v = v + self.dt * dv_dt
            w = w + self.dt * dw_dt

        return torch.stack([v, w], dim=1)


class PMNNBlock(nn.Module):
    """
    PMNN block using an FHN solver as a nonlinear dynamical hidden layer.

    This version is adapted for the Energy Efficiency regression dataset.

    Data interpretation:

        Original tabular input:
            x.shape = (batch_size, 8)

        Regression target:
            y.shape = (batch_size, 2)

        The two target variables are:

            1. Heating Load
            2. Cooling Load

    Architecture:

        input vector of 8 features
            -> Linear(input_size, hidden_size)
            -> LayerNorm(hidden_size)
            -> activation
            -> extract two components as FHN initial state: v0 and w0
            -> FitzHugh-Nagumo solver
            -> Linear(2, output_size)

    For Energy Efficiency regression:

        input_size = 8
        hidden_size >= 2
        output_size = 2

    The model returns raw continuous predictions. Do not apply sigmoid,
    softmax or argmax. Use torch.nn.MSELoss in the training script.
    """

    def __init__(
        self,
        *,
        input_size: int = 8,
        hidden_size: int = 2,
        output_size: int = 2,
        activation=nn.SiLU,
        dt: int = 5,
        t_end: int = 10,
        a: float = 0.2,
        b: float = 0.02,
        g: float = 3.0,
        I: float = 0.0,
        device=None,
        dtype=None,
    ):
        """
        Initializer for `PMNNBlock`.

        Inputs:
            * input_size: number of input features.
              For Energy Efficiency, use 8.
            * hidden_size: size of the linear embedding producing FHN initial state.
              It must be at least 2 because the FHN solver needs v0 and w0.
            * output_size: number of output variables.
              For Energy Efficiency with Heating Load and Cooling Load, use 2.
            * activation: activation function class for the pre-dynamical embedding.
            * dt: integration step for FHN solver.
            * t_end: total simulated time for FHN solver.
            * a, b, g, I: parameters forwarded to `FitzHughNagumoSolver`.
            * device
            * dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}.")

        if self.hidden_size < 2:
            raise ValueError(
                "hidden_size must be at least 2 because the FHN solver needs "
                "two initial conditions: v0 and w0."
            )

        if self.output_size <= 0:
            raise ValueError(f"output_size must be positive, got {self.output_size}.")

        self.hidden = nn.Linear(
            self.input_size,
            self.hidden_size,
            bias=True,
            **factory_kwargs,
        )

        self.hidden_norm = nn.LayerNorm(
            self.hidden_size,
            **factory_kwargs,
        )

        self.hidden_act = activation()

        self.solver = FitzHughNagumoSolver(
            dt=dt,
            t_end=t_end,
            a=a,
            b=b,
            g=g,
            I=I,
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
        Forward method for `PMNNBlock`.

        Inputs:
            * x: input tensor of shape (batch_size, 8).

        Outputs:
            * y: output tensor of shape (batch_size, 2).

        The two output components correspond to the two numerical targets:

            y[:, 0] -> Heating Load
            y[:, 1] -> Cooling Load
        """
        if x.dim() == 1:
            x = x.view(1, -1)

        if x.dim() != 2:
            x = x.view(x.size(0), -1)

        if x.size(1) != self.input_size:
            raise ValueError(
                f"Expected input with {self.input_size} features, "
                f"but received {x.size(1)}."
            )

        h = self.hidden_act(self.hidden_norm(self.hidden(x)))

        v0 = h[:, 0]
        w0 = h[:, 1]

        terminal_state = self.solver(v0, w0)

        y = self.output(terminal_state)

        return y

    def save(self, file):
        """
        Save block state.

        Inputs:
            * file: path to state file.
        """
        torch.save(deepcopy(self.state_dict()), file)

    def load(self, file):
        """
        Load block state.

        Inputs:
            * file: path to state file.
        """
        self.load_state_dict(torch.load(file, weights_only=True))

