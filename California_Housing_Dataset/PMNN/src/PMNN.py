import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FitzHughNagumoSolver(nn.Module):
    """
    Discrete-time FitzHugh–Nagumo (FHN) ODE solver module.

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

    The block maps an input vector `x` to a hidden embedding, extracts two components
    as initial conditions `(v0, w0)` for the `FitzHughNagumoSolver`, integrates the
    dynamics, and finally projects the terminal state `[v_T, w_T]` to the desired output.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
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
            * hidden_size: size of the linear embedding producing FHN initial state.
            * output_size: number of output features.
            * activation: activation function class for the pre-dynamical embedding.
            * dt: integration step for FHN solver.
            * t_end: total simulated time for FHN solver.
            * a, b, g, I: parameters forwarded to `FitzHughNagumoSolver`.
            * device
            * dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.hidden = nn.Linear(input_size, hidden_size, bias=True, **factory_kwargs)
        self.hidden_norm = nn.LayerNorm(hidden_size, **factory_kwargs)
        self.hidden_act = activation()

        self.solver = FitzHughNagumoSolver(
            dt=dt, t_end=t_end, a=a, b=b, g=g, I=I, **factory_kwargs
        )

        self.output = nn.Linear(2, output_size, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for `PMNNBlock`.

        Inputs:
            * x: input tensor of shape (N, input_size).

        Outputs:
            * Output tensor of shape (N, output_size).
        """
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
