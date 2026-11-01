"""A from-scratch multi-layer feedforward network trained with back-propagation.

The implementation follows the assignment's notation:

* ``net = W @ input + b``
* output delta: ``(target - output) * f'(net)``
* hidden delta: ``f'(net) * (W_next.T @ delta_next)``
* update: ``W_new = W_old + eta * delta @ input.T`` (and ``b += eta * delta``)

Internally the equivalent gradient form ``parameter -= learning_rate *
gradient`` is used, where the gradient of ``0.5 * sum((target - output)^2)``
equals ``-delta @ input.T``. This makes numerical gradient checking
straightforward while preserving the professor's update rule exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.activations import get_activation
from src.encoding import CLASS_NAMES, FEATURE_NAMES


class BackPropagationNetwork:
    """Configurable fully connected network with one hidden-layer stack."""

    def __init__(
        self,
        hidden_layers: Sequence[int] = (8,),
        learning_rate: float = 0.05,
        epochs: int = 1000,
        use_bias: bool = True,
        activation: str = "sigmoid",
        random_state: int | None = 42,
    ):
        self.hidden_layers = self._validate_hidden_layers(hidden_layers)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be a finite value greater than 0.")
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)):
            raise ValueError("Epochs must be an integer greater than 0.")
        if int(epochs) <= 0:
            raise ValueError("Epochs must be an integer greater than 0.")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, (int, np.integer))
        ):
            raise ValueError("Random seed must be an integer or None.")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.use_bias = bool(use_bias)
        self.random_state = None if random_state is None else int(random_state)
        (
            self.activation_name,
            self._activation,
            self._activation_derivative,
        ) = get_activation(activation)

        self.n_inputs = len(FEATURE_NAMES)
        self.n_outputs = len(CLASS_NAMES)
        self.layer_sizes: tuple[int, ...] = (
            self.n_inputs,
            *self.hidden_layers,
            self.n_outputs,
        )

        self._rng = np.random.default_rng(self.random_state)
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray | None] = []
        self.loss_history_: list[float] = []
        self._initialize_parameters()

    @staticmethod
    def _validate_hidden_layers(hidden_layers: Sequence[int]) -> tuple[int, ...]:
        if isinstance(hidden_layers, (str, bytes)):
            raise ValueError("Hidden layers must be a sequence of positive integers.")
        try:
            values = tuple(hidden_layers)
        except TypeError as exc:
            raise ValueError(
                "Hidden layers must be a sequence of positive integers."
            ) from exc
        if not values:
            raise ValueError("At least one hidden layer is required.")
        for width in values:
            if isinstance(width, bool) or not isinstance(width, (int, np.integer)):
                raise ValueError("Every hidden-layer width must be an integer.")
            if int(width) <= 0:
                raise ValueError("Every hidden-layer width must be greater than 0.")
        return tuple(int(width) for width in values)

    def _initialize_parameters(self) -> None:
        """Initialize weights and biases with small random values."""
        self._rng = np.random.default_rng(self.random_state)
        self.weights_ = []
        self.biases_ = []
        for inputs, outputs in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights_.append(
                self._rng.normal(loc=0.0, scale=0.1, size=(outputs, inputs))
            )
            self.biases_.append(
                self._rng.normal(loc=0.0, scale=0.1, size=outputs)
                if self.use_bias
                else None
            )

    def _forward(
        self, sample: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return every layer activation and pre-activation for one sample."""
        activations: list[np.ndarray] = [sample]
        nets: list[np.ndarray] = []
        current = sample
        for weights, bias in zip(self.weights_, self.biases_):
            net = weights @ current
            if bias is not None:
                net = net + bias
            nets.append(net)
            current = self._activation(net)
            activations.append(current)
        return activations, nets

    def forward(self, sample: np.ndarray) -> np.ndarray:
        """Return the three output-neuron activations for one sample."""
        values = self._validate_single_sample(sample)
        activations, _ = self._forward(values)
        return activations[-1].copy()

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """Return output activations for one or more samples.

        For sigmoid these lie in (0, 1), but they are not calibrated
        probabilities. For tanh they lie in (-1, 1). They are used only as
        class scores and decoded with ``argmax``.
        """
        matrix = np.asarray(samples, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected samples with shape (n, {self.n_inputs}); "
                f"got {matrix.shape}."
            )
        return np.vstack([self.forward(row) for row in matrix])

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Return the predicted class index (0..2) for each sample."""
        return np.argmax(self.predict_proba(samples), axis=1).astype(int)

    def loss_and_gradients(
        self, sample: np.ndarray, target: np.ndarray
    ) -> tuple[float, list[np.ndarray], list[np.ndarray | None]]:
        """Return half-SSE loss and gradients for one sample.

        ``loss = 0.5 * sum((target - output)^2)``. The gradients are the true
        derivatives of that loss with respect to every weight and bias, so a
        numerical gradient check can validate the back-propagation code.
        All deltas are computed before any parameter is changed.
        """
        values = self._validate_single_sample(sample)
        desired = np.asarray(target, dtype=float).reshape(-1)
        if desired.shape[0] != self.n_outputs:
            raise ValueError(
                f"Target must have {self.n_outputs} values; got {desired.shape[0]}."
            )

        activations, nets = self._forward(values)
        output = activations[-1]
        loss = float(0.5 * np.sum((desired - output) ** 2))

        deltas: list[np.ndarray] = [np.empty(0)] * len(self.weights_)
        deltas[-1] = (desired - output) * self._activation_derivative(nets[-1])

        for layer in range(len(self.weights_) - 2, -1, -1):
            next_weights = self.weights_[layer + 1]
            deltas[layer] = (
                self._activation_derivative(nets[layer])
                * (next_weights.T @ deltas[layer + 1])
            )

        weight_gradients: list[np.ndarray] = []
        bias_gradients: list[np.ndarray | None] = []
        for layer, delta in enumerate(deltas):
            # d(0.5*SSE)/dW = -(delta @ input.T)
            weight_gradients.append(-np.outer(delta, activations[layer]))
            bias_gradients.append(-delta.copy() if self.use_bias else None)

        return loss, weight_gradients, bias_gradients

    def fit(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True,
    ) -> "BackPropagationNetwork":
        """Train with online/stochastic gradient descent for all epochs."""
        from src.encoding import one_hot_encode

        matrix = np.asarray(samples, dtype=float)
        targets = np.asarray(labels)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_inputs:
            raise ValueError(
                f"Training data must have shape (n, {self.n_inputs}); "
                f"got {matrix.shape}."
            )
        if targets.ndim != 1 or len(targets) != len(matrix):
            raise ValueError("Labels must be a 1D array with one label per sample.")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Labels must be integer class indices.")
        if targets.size and (targets.min() < 0 or targets.max() >= self.n_outputs):
            raise ValueError(f"Labels must lie in 0..{self.n_outputs - 1}.")
        if len(matrix) == 0:
            raise ValueError("Training data must not be empty.")

        self._initialize_parameters()
        self.loss_history_ = []
        desired_outputs = one_hot_encode(targets, self.n_outputs)
        number_of_samples = len(matrix)

        for _epoch in range(self.epochs):
            order = (
                self._rng.permutation(number_of_samples)
                if shuffle
                else np.arange(number_of_samples)
            )
            squared_error_sum = 0.0
            for index in order:
                target = desired_outputs[index]
                half_sse, weight_gradients, bias_gradients = (
                    self.loss_and_gradients(matrix[index], target)
                )

                # Equivalent to W += eta * delta @ input.T and b += eta * delta.
                for layer in range(len(self.weights_)):
                    self.weights_[layer] -= (
                        self.learning_rate * weight_gradients[layer]
                    )
                    if self.biases_[layer] is not None:
                        self.biases_[layer] -= (
                            self.learning_rate * bias_gradients[layer]
                        )

                # half-SSE = 0.5*sum(error^2); per-sample MSE = 2*half-SSE/n_out.
                squared_error_sum += 2.0 * half_sse / self.n_outputs

            epoch_mse = squared_error_sum / number_of_samples
            self.loss_history_.append(float(epoch_mse))
            if not np.isfinite(epoch_mse):
                raise FloatingPointError(
                    "Training diverged (non-finite loss). Reduce the learning rate."
                )

        if not all(np.isfinite(weights).all() for weights in self.weights_):
            raise FloatingPointError(
                "Training produced non-finite weights. Reduce the learning rate."
            )
        return self

    def _validate_single_sample(self, sample: np.ndarray) -> np.ndarray:
        values = np.asarray(sample, dtype=float).reshape(-1)
        if values.shape[0] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input features; got {values.shape[0]}."
            )
        if not np.isfinite(values).all():
            raise ValueError("Input features must all be finite numbers.")
        return values

    @property
    def architecture_description(self) -> str:
        """Human-readable layer description, for example ``5-8-3``."""
        return "-".join(str(size) for size in self.layer_sizes)
