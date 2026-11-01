"""A small, from-scratch binary Single-Layer Perceptron classifier."""

import numpy as np


class Perceptron:
    """Perceptron trained with the classic online error-correction rule.

    Labels must be encoded as -1 and +1. The model intentionally supports
    exactly two input features, as required by the Lab 3 task.

    For a misclassified sample, the update is:
        delta = learning_rate * (target - prediction)
        weights += delta * features
        bias += delta  (only when use_bias=True)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        use_bias: bool = True,
        random_state: int | None = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.use_bias = bool(use_bias)
        self.random_state = random_state

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.errors_: list[int] = []
        self.n_features_in_: int | None = None

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (samples, 2).")
        if X.shape[1] != 2:
            raise ValueError("Perceptron expects exactly two features.")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values; preprocess it first.")
        return X

    @staticmethod
    def _validate_y(y: np.ndarray, n_samples: int) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != n_samples:
            raise ValueError("y must be a 1D array with one label per sample.")
        if not np.isin(y, [-1, 1]).all():
            raise ValueError("Labels must be encoded as -1 and +1.")
        return y.astype(int)

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Calculate weighted input: X @ weights + bias."""
        if self.weights_ is None:
            raise RuntimeError("The model must be fitted before calling net_input().")
        X = self._validate_X(X)
        return X @ self.weights_ + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict -1 or +1; a net input of exactly zero maps to +1."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Initialize small random parameters and train sample by sample."""
        X = self._validate_X(X)
        y = self._validate_y(y, X.shape[0])

        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.uniform(-0.01, 0.01, size=X.shape[1])
        self.bias_ = float(rng.uniform(-0.01, 0.01)) if self.use_bias else 0.0
        self.n_features_in_ = X.shape[1]
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for features, target in zip(X, y):
                net = float(np.dot(features, self.weights_) + self.bias_)
                prediction = 1 if net >= 0.0 else -1

                if prediction != target:
                    delta = self.learning_rate * (target - prediction)
                    self.weights_ += delta * features
                    if self.use_bias:
                        self.bias_ += delta
                    errors += 1

            self.errors_.append(errors)
            # Stop early when every training sample is classified correctly.
            if errors == 0:
                break

        return self
