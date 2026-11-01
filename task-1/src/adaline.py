"""A from-scratch ADALINE classifier trained with mean squared error."""

import numpy as np


class Adaline:
    """Adaptive Linear Neuron for binary classification.

    The linear output is trained against targets -1/+1 using batch gradient
    descent on mean squared error. Prediction converts the linear output to
    the same -1/+1 labels used by the Perceptron.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        mse_threshold: float = 0.01,
        use_bias: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if mse_threshold < 0:
            raise ValueError("mse_threshold cannot be negative.")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.mse_threshold = float(mse_threshold)
        self.use_bias = bool(use_bias)
        self.random_state = random_state
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.mse_history_: list[float] = []

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("ADALINE expects X with exactly two features.")
        if X.shape[0] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be non-empty and contain finite values only.")
        return X

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != n_samples:
            raise ValueError("y must be a 1D array with one label per sample.")
        if not np.isin(y, [-1, 1]).all():
            raise ValueError("Labels must be encoded as -1 and +1.")
        return y.astype(float)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("The model must be fitted before prediction.")
        X = self._validate_X(X)
        return X @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify linear outputs at threshold zero."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Adaline":
        """Train with batch gradient descent on MSE."""
        X = self._validate_X(X)
        y = self._validate_y(y, len(X))
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.uniform(-0.01, 0.01, size=X.shape[1])
        self.bias_ = float(rng.uniform(-0.01, 0.01)) if self.use_bias else 0.0
        self.mse_history_ = []
        n_samples = len(X)

        for _ in range(self.epochs):
            outputs = X @ self.weights_ + self.bias_
            errors = y - outputs
            mse = float(np.mean(errors ** 2))
            self.mse_history_.append(mse)

            # Batch gradient descent: update after measuring all samples.
            self.weights_ += self.learning_rate * (X.T @ errors) / n_samples
            if self.use_bias:
                self.bias_ += self.learning_rate * float(np.mean(errors))

            if mse <= self.mse_threshold:
                break

        # Record the final post-update MSE so the history reflects the final parameters.
        final_errors = y - (X @ self.weights_ + self.bias_)
        self.final_mse_ = float(np.mean(final_errors ** 2))
        return self
