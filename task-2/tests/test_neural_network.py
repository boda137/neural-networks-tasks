import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encoding import encode_labels, one_hot_encode
from src.neural_network import BackPropagationNetwork

ROOT = Path(__file__).resolve().parents[1]


def _small_dataset(seed: int = 7):
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [1.5, 1.5, 1.5, 1.5, 1.5],
            [-1.5, -1.5, -1.5, -1.5, -1.5],
            [1.5, -1.5, 0.0, 1.5, -1.5],
        ]
    )
    samples = np.vstack(
        [center + rng.normal(0.0, 0.15, size=(8, 5)) for center in centers]
    )
    labels = np.repeat([0, 1, 2], 8)
    return samples, labels


def test_architecture_dimensions_for_multiple_hidden_layers():
    model = BackPropagationNetwork(hidden_layers=(4, 3), random_state=1)
    assert model.layer_sizes == (5, 4, 3, 3)
    assert [weights.shape for weights in model.weights_] == [
        (4, 5),
        (3, 4),
        (3, 3),
    ]
    assert [bias.shape for bias in model.biases_] == [(4,), (3,), (3,)]


def test_different_neuron_counts_are_supported():
    model = BackPropagationNetwork(hidden_layers=(2, 7, 1), random_state=2)
    assert model.layer_sizes == (5, 2, 7, 1, 3)
    assert model.architecture_description == "5-2-7-1-3"


@pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
@pytest.mark.parametrize("use_bias", [True, False])
def test_forward_returns_three_outputs(activation, use_bias):
    model = BackPropagationNetwork(
        hidden_layers=(5, 3),
        activation=activation,
        use_bias=use_bias,
        random_state=3,
    )
    output = model.forward(np.zeros(5))
    assert output.shape == (3,)
    assert np.isfinite(output).all()
    if activation == "sigmoid":
        assert np.all((output > 0) & (output < 1))
    else:
        assert np.all((output >= -1) & (output <= 1))


def test_bias_can_be_disabled():
    model = BackPropagationNetwork(
        hidden_layers=(4,), use_bias=False, random_state=4
    )
    assert all(bias is None for bias in model.biases_)
    samples, labels = _small_dataset()
    before = [weights.copy() for weights in model.weights_]
    model.fit(samples, labels, shuffle=False)
    assert all(bias is None for bias in model.biases_)
    assert any(
        not np.allclose(old, new)
        for old, new in zip(before, model.weights_)
    )


def test_predictions_return_valid_class_ids():
    samples, labels = _small_dataset()
    model = BackPropagationNetwork(
        hidden_layers=(6,), epochs=20, random_state=5
    ).fit(samples, labels)
    predictions = model.predict(samples)
    assert predictions.shape == (len(samples),)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert set(predictions).issubset({0, 1, 2})
    probabilities = model.predict_proba(samples)
    assert probabilities.shape == (len(samples), 3)


def test_weights_change_during_training():
    samples, labels = _small_dataset()
    model = BackPropagationNetwork(
        hidden_layers=(5,), epochs=5, random_state=6
    )
    before = [weights.copy() for weights in model.weights_]
    model.fit(samples, labels)
    assert any(
        not np.allclose(old, new)
        for old, new in zip(before, model.weights_)
    )
    assert len(model.loss_history_) == model.epochs


def test_loss_decreases_on_small_controlled_dataset():
    samples, labels = _small_dataset()
    model = BackPropagationNetwork(
        hidden_layers=(10,),
        learning_rate=0.05,
        epochs=600,
        activation="sigmoid",
        random_state=8,
    ).fit(samples, labels)
    assert model.loss_history_[-1] < model.loss_history_[0] * 0.5
    training_accuracy = np.mean(model.predict(samples) == labels)
    assert training_accuracy >= 0.9


def test_same_seed_is_reproducible():
    samples, labels = _small_dataset()
    first = BackPropagationNetwork(
        hidden_layers=(4, 3), epochs=30, random_state=11
    )
    second = BackPropagationNetwork(
        hidden_layers=(4, 3), epochs=30, random_state=11
    )
    for first_weights, second_weights in zip(first.weights_, second.weights_):
        np.testing.assert_array_equal(first_weights, second_weights)
    first.fit(samples, labels)
    second.fit(samples, labels)
    np.testing.assert_allclose(first.loss_history_, second.loss_history_)
    for first_weights, second_weights in zip(first.weights_, second.weights_):
        np.testing.assert_allclose(first_weights, second_weights)
    np.testing.assert_array_equal(first.predict(samples), second.predict(samples))


@pytest.mark.parametrize("activation", ["sigmoid", "tanh"])
def test_backpropagation_gradients_match_numerical_gradients(activation):
    model = BackPropagationNetwork(
        hidden_layers=(3,),
        learning_rate=0.05,
        epochs=1,
        activation=activation,
        random_state=13,
    )
    sample = np.array([0.4, -0.7, 0.2, 0.9, -0.3])
    target = one_hot_encode([1])[0]
    _loss, weight_gradients, bias_gradients = model.loss_and_gradients(
        sample, target
    )
    epsilon = 1e-6

    for layer, weights in enumerate(model.weights_):
        for index in np.ndindex(weights.shape):
            original = weights[index]
            weights[index] = original + epsilon
            loss_plus, _, _ = model.loss_and_gradients(sample, target)
            weights[index] = original - epsilon
            loss_minus, _, _ = model.loss_and_gradients(sample, target)
            weights[index] = original
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            assert numerical == pytest.approx(
                weight_gradients[layer][index], rel=1e-5, abs=1e-7
            )

    for layer, bias in enumerate(model.biases_):
        assert bias is not None
        for index in range(len(bias)):
            original = bias[index]
            bias[index] = original + epsilon
            loss_plus, _, _ = model.loss_and_gradients(sample, target)
            bias[index] = original - epsilon
            loss_minus, _, _ = model.loss_and_gradients(sample, target)
            bias[index] = original
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            assert numerical == pytest.approx(
                bias_gradients[layer][index], rel=1e-5, abs=1e-7
            )


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError):
        BackPropagationNetwork(hidden_layers=())
    with pytest.raises(ValueError):
        BackPropagationNetwork(hidden_layers=(0,))
    with pytest.raises(ValueError):
        BackPropagationNetwork(hidden_layers=(4,), learning_rate=0)
    with pytest.raises(ValueError):
        BackPropagationNetwork(hidden_layers=(4,), epochs=0)
    with pytest.raises(ValueError):
        BackPropagationNetwork(hidden_layers=(4,), activation="relu")


def test_fit_validates_shapes_and_labels():
    samples, labels = _small_dataset()
    model = BackPropagationNetwork(hidden_layers=(4,), epochs=1)
    with pytest.raises(ValueError):
        model.fit(samples[:, :4], labels)
    with pytest.raises(ValueError):
        model.fit(samples, labels[:-1])
    invalid_labels = labels.copy()
    invalid_labels[0] = 5
    with pytest.raises(ValueError):
        model.fit(samples, invalid_labels)
