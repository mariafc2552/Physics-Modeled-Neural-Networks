import logging
from copy import deepcopy
from typing import Optional, Sequence, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DNNBlock(nn.Module):
    """
    Fully connected feed-forward neural network for multi-class classification.

    This version is adapted for the Iris dataset.

    The architecture is defined by the `hidden_sizes` argument, which allows the
    number of hidden layers and the number of neurons per layer to be controlled
    externally from the configuration file or the hyperparameter search routine.

    Examples:
        hidden_sizes=(64, 32, 16) -> three hidden layers
        hidden_sizes=(128, 64)    -> two hidden layers
        hidden_sizes=(256,)       -> one hidden layer

    For Iris:
        input_size = 4
        output_size = 3

    The model returns raw logits. Do not apply softmax inside the model.
    Use CrossEntropyLoss in the training script.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        activation: Type[nn.Module] = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        use_layer_norm: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError(f"input_size must be positive. Received: {input_size}")

        if input_size != 4:
            raise ValueError(
                "input_size must be 4 for the Iris dataset."
            )

        if output_size <= 0:
            raise ValueError(f"output_size must be positive. Received: {output_size}")

        if output_size != 3:
            raise ValueError(
                "output_size must be 3 for Iris multi-class classification "
                "with CrossEntropyLoss."
            )

        if hidden_sizes is None or len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one hidden layer.")

        hidden_sizes = tuple(int(h) for h in hidden_sizes)

        if any(h <= 0 for h in hidden_sizes):
            raise ValueError(
                f"All hidden layer sizes must be positive. Received: {hidden_sizes}"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        activation_kwargs = activation_kwargs or {}

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.use_layer_norm = use_layer_norm
        self.bias = bias

        layers = []
        previous_size = input_size

        for layer_index, hidden_size in enumerate(self.hidden_sizes):
            layers.append(
                nn.Linear(
                    previous_size,
                    hidden_size,
                    bias=bias,
                    **factory_kwargs,
                )
            )

            if use_layer_norm:
                layers.append(
                    nn.LayerNorm(
                        hidden_size,
                        **factory_kwargs,
                    )
                )

            layers.append(activation(**activation_kwargs))
            previous_size = hidden_size

        self.hidden = nn.Sequential(*layers)

        self.output = nn.Linear(
            previous_size,
            output_size,
            bias=bias,
            **factory_kwargs,
        )

        logger.info(
            "Initialized DNNBlock with input_size=%s, hidden_sizes=%s, "
            "output_size=%s, activation=%s, use_layer_norm=%s",
            input_size,
            self.hidden_sizes,
            output_size,
            activation.__name__,
            use_layer_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        y = self.output(h)
        return y

    def count_trainable_parameters(self) -> int:
        """
        Return the number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_name(self) -> str:
        """
        Return a compact string representation of the architecture.
        """
        hidden = "-".join(str(h) for h in self.hidden_sizes)
        return f"DNN_{self.input_size}_{hidden}_{self.output_size}"

    def save(self, file):
        torch.save(deepcopy(self.state_dict()), file)

    def load(self, file, map_location=None):
        state_dict = torch.load(
            file,
            map_location=map_location,
            weights_only=True,
        )
        self.load_state_dict(state_dict)