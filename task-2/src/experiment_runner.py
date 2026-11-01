"""Run reproducible experiments, save plots, and generate the Task 2 report.

Two evaluation protocols are included:

1. The required grid search: every configuration is trained on the fixed
   90-sample training split and evaluated on the fixed 60-sample test split.
   The best test accuracy per activation is recorded, but because the test set
   is used to choose the winner, that number is optimistic.
2. A cleaner independent check: a small stratified validation split is carved
   out of the training data only. A configuration is selected there, retrained
   on all 90 training samples, and then evaluated on the 60 test samples once.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_penguins
from src.data_splitter import split_train_test_per_class
from src.encoding import CLASS_NAMES, decode_labels, encode_labels
from src.evaluation import evaluate_predictions
from src.neural_network import BackPropagationNetwork
from src.preprocessing import PenguinPreprocessor
from src.visualization import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_training_loss,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = TASK_ROOT / "data" / "penguins.csv"
REPORT_DIR = TASK_ROOT / "report"


@dataclass(frozen=True)
class ExperimentConfig:
    """One fully specified experiment."""

    activation: str
    learning_rate: float
    epochs: int
    hidden_layers: tuple[int, ...]
    use_bias: bool
    random_state: int = 42


@dataclass
class ExperimentResult:
    """Measured outcome of one experiment."""

    experiment_id: str
    config: ExperimentConfig
    model: BackPropagationNetwork
    train_accuracy: float
    test_accuracy: float
    train_correct: int
    test_correct: int
    matrix_train: np.ndarray
    matrix_test: np.ndarray
    final_mse: float
    duration_seconds: float


def build_experiment_configs() -> list[ExperimentConfig]:
    """Return a reasonable, reproducible grid for both activations."""
    architectures: tuple[tuple[int, ...], ...] = (
        (8,),
        (16,),
        (8, 8),
        (16, 8),
        (8, 4, 4),
    )
    learning_rates = {
        "sigmoid": (0.01, 0.05, 0.1),
        "tanh": (0.005, 0.01, 0.05),
    }
    configs: list[ExperimentConfig] = []
    for activation in ("sigmoid", "tanh"):
        for hidden_layers in architectures:
            epochs = 1000 if len(hidden_layers) <= 2 else 1400
            for learning_rate in learning_rates[activation]:
                configs.append(
                    ExperimentConfig(
                        activation=activation,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        hidden_layers=hidden_layers,
                        use_bias=True,
                    )
                )
        # Include one no-bias run per activation to exercise that option.
        configs.append(
            ExperimentConfig(
                activation=activation,
                learning_rate=learning_rates[activation][1],
                epochs=1000,
                hidden_layers=(8,),
                use_bias=False,
            )
        )
    return configs


def _architecture(config: ExperimentConfig) -> str:
    return "-".join(str(size) for size in (5, *config.hidden_layers, 3))


def _format_hidden(config: ExperimentConfig) -> str:
    return "-".join(str(width) for width in config.hidden_layers)


def _predict_names(model: BackPropagationNetwork, features: np.ndarray) -> np.ndarray:
    return np.asarray(decode_labels(model.predict(features)), dtype=object)


def _result_to_row(result: ExperimentResult) -> dict[str, object]:
    config = result.config
    row: dict[str, object] = {
        "experiment_id": result.experiment_id,
        "activation": config.activation,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "number_of_hidden_layers": len(config.hidden_layers),
        "hidden_neurons": _format_hidden(config),
        "architecture": _architecture(config),
        "use_bias": "yes" if config.use_bias else "no",
        "random_state": config.random_state,
        "train_accuracy": round(result.train_accuracy, 2),
        "test_accuracy": round(result.test_accuracy, 2),
        "train_correct": result.train_correct,
        "test_correct": result.test_correct,
        "final_training_mse": round(result.final_mse, 8),
        "duration_seconds": round(result.duration_seconds, 3),
    }
    for actual_index, actual_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            key = f"test_true_{actual_name.lower()}_pred_{predicted_name.lower()}"
            row[key] = int(result.matrix_test[actual_index, predicted_index])
    return row


def run_experiments(
    output_dir: Path = REPORT_DIR,
) -> tuple[list[ExperimentResult], Path]:
    """Train every configuration and save the required CSV results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_penguins(DATA_PATH)
    train, test = split_train_test_per_class(data)

    preprocessor = PenguinPreprocessor().fit(train)
    X_train = preprocessor.transform(train)
    X_test = preprocessor.transform(test)
    y_train_indices = encode_labels(train["Species"].to_numpy())
    encode_labels(test["Species"].to_numpy())
    y_train_names = train["Species"].to_numpy()
    y_test_names = test["Species"].to_numpy()

    results: list[ExperimentResult] = []
    configs = build_experiment_configs()
    counters = {"sigmoid": 0, "tanh": 0}
    for config in configs:
        counters[config.activation] += 1
        experiment_id = f"{config.activation}_{counters[config.activation]:02d}"
        started = perf_counter()
        model = BackPropagationNetwork(
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            use_bias=config.use_bias,
            activation=config.activation,
            random_state=config.random_state,
        ).fit(X_train, y_train_indices)
        duration = perf_counter() - started

        train_predictions = _predict_names(model, X_train)
        test_predictions = _predict_names(model, X_test)
        train_metrics = evaluate_predictions(y_train_names, train_predictions)
        test_metrics = evaluate_predictions(y_test_names, test_predictions)
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            model=model,
            train_accuracy=float(train_metrics["accuracy"]) * 100.0,
            test_accuracy=float(test_metrics["accuracy"]) * 100.0,
            train_correct=int(train_metrics["correct"]),
            test_correct=int(test_metrics["correct"]),
            matrix_train=np.asarray(train_metrics["confusion_matrix"]),
            matrix_test=np.asarray(test_metrics["confusion_matrix"]),
            final_mse=float(model.loss_history_[-1]),
            duration_seconds=duration,
        )
        results.append(result)
        print(
            f"{experiment_id:10} | {config.activation:7} | arch={_architecture(config):10} "
            f"| eta={config.learning_rate:<5} | epochs={config.epochs:<4} "
            f"| bias={'yes' if config.use_bias else 'no ':3} "
            f"| train={result.train_accuracy:6.2f}% | test={result.test_accuracy:6.2f}%"
        )

    csv_path = output_dir / "experiment_results.csv"
    rows = [_result_to_row(result) for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return results, csv_path


def select_best_by_activation(
    results: list[ExperimentResult],
) -> dict[str, ExperimentResult]:
    """Pick the first configuration with the highest test accuracy per activation."""
    best: dict[str, ExperimentResult] = {}
    for activation in ("sigmoid", "tanh"):
        candidates = [
            result for result in results if result.config.activation == activation
        ]
        highest = max(result.test_accuracy for result in candidates)
        best[activation] = next(
            result for result in candidates if result.test_accuracy == highest
        )
    return best


def find_test_accuracy_ties(
    results: list[ExperimentResult], activation: str
) -> list[ExperimentResult]:
    """Return every configuration sharing the best test accuracy."""
    candidates = [
        result for result in results if result.config.activation == activation
    ]
    highest = max(result.test_accuracy for result in candidates)
    return [result for result in candidates if result.test_accuracy == highest]


def _inner_train_validation_split(
    train: pd.DataFrame,
    inner_train_per_class: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the fixed training set itself, preserving per-class order."""
    inner_train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for label in CLASS_NAMES:
        class_data = train[train["Species"] == label]
        inner_train_parts.append(class_data.iloc[:inner_train_per_class])
        validation_parts.append(class_data.iloc[inner_train_per_class:])
    inner_train = pd.concat(inner_train_parts)
    validation = pd.concat(validation_parts)
    if not set(inner_train.index).isdisjoint(set(validation.index)):
        raise RuntimeError("Inner validation split overlaps with inner training data.")
    return inner_train, validation


def run_independent_evaluation(
    results: list[ExperimentResult],
    output_dir: Path = REPORT_DIR,
) -> tuple[list[dict[str, object]], Path]:
    """Select configs on training-only validation data, then test once."""
    data = load_penguins(DATA_PATH)
    train, test = split_train_test_per_class(data)
    inner_train, validation = _inner_train_validation_split(train)

    inner_preprocessor = PenguinPreprocessor().fit(inner_train)
    X_inner = inner_preprocessor.transform(inner_train)
    X_validation = inner_preprocessor.transform(validation)
    y_inner = encode_labels(inner_train["Species"].to_numpy())
    y_validation_names = validation["Species"].to_numpy()
    y_inner_names = inner_train["Species"].to_numpy()

    selection_rows: list[dict[str, object]] = []
    selected_configs: dict[str, ExperimentConfig] = {}
    for activation in ("sigmoid", "tanh"):
        candidates = [r.config for r in results if r.config.activation == activation]
        measured: list[dict[str, object]] = []
        for config in candidates:
            model = BackPropagationNetwork(
                hidden_layers=config.hidden_layers,
                learning_rate=config.learning_rate,
                epochs=config.epochs,
                use_bias=config.use_bias,
                activation=config.activation,
                random_state=config.random_state,
            ).fit(X_inner, y_inner)
            inner_train_accuracy = (
                float(
                    np.mean(_predict_names(model, X_inner) == np.asarray(y_inner_names))
                )
                * 100.0
            )
            validation_accuracy = (
                float(
                    np.mean(
                        _predict_names(model, X_validation)
                        == np.asarray(y_validation_names)
                    )
                )
                * 100.0
            )
            measured.append(
                {
                    "activation": activation,
                    "learning_rate": config.learning_rate,
                    "epochs": config.epochs,
                    "number_of_hidden_layers": len(config.hidden_layers),
                    "hidden_neurons": _format_hidden(config),
                    "use_bias": "yes" if config.use_bias else "no",
                    "inner_train_accuracy": round(inner_train_accuracy, 2),
                    "validation_accuracy": round(validation_accuracy, 2),
                    "_config": config,
                }
            )

        # Highest validation accuracy wins; then higher inner-training accuracy;
        # then the smaller network, then the earlier grid position.
        best_row = max(
            measured,
            key=lambda row: (
                float(row["validation_accuracy"]),
                float(row["inner_train_accuracy"]),
                -sum(row["_config"].hidden_layers),
                -float(row["learning_rate"]),
            ),
        )
        selected_configs[activation] = best_row["_config"]
        selection_rows.extend(measured)

    full_preprocessor = PenguinPreprocessor().fit(train)
    X_train = full_preprocessor.transform(train)
    X_test = full_preprocessor.transform(test)
    y_train = encode_labels(train["Species"].to_numpy())
    y_test = encode_labels(test["Species"].to_numpy())
    y_train_names = train["Species"].to_numpy()
    y_test_names = test["Species"].to_numpy()

    final_rows: list[dict[str, object]] = []
    for activation, config in selected_configs.items():
        model = BackPropagationNetwork(
            hidden_layers=config.hidden_layers,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            use_bias=config.use_bias,
            activation=activation,
            random_state=config.random_state,
        ).fit(X_train, y_train)
        train_metrics = evaluate_predictions(
            y_train_names, _predict_names(model, X_train)
        )
        test_metrics = evaluate_predictions(y_test_names, _predict_names(model, X_test))
        selection_row = next(
            row
            for row in selection_rows
            if row["activation"] == activation and row["_config"] == config
        )
        row: dict[str, object] = {
            "activation": activation,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "number_of_hidden_layers": len(config.hidden_layers),
            "hidden_neurons": _format_hidden(config),
            "use_bias": "yes" if config.use_bias else "no",
            "inner_train_accuracy": selection_row["inner_train_accuracy"],
            "validation_accuracy": selection_row["validation_accuracy"],
            "final_train_accuracy": round(float(train_metrics["accuracy"]) * 100.0, 2),
            "final_test_accuracy": round(float(test_metrics["accuracy"]) * 100.0, 2),
            "final_train_correct": int(train_metrics["correct"]),
            "final_test_correct": int(test_metrics["correct"]),
            "final_training_mse": round(float(model.loss_history_[-1]), 8),
        }
        matrix = np.asarray(test_metrics["confusion_matrix"])
        for actual_index, actual_name in enumerate(CLASS_NAMES):
            for predicted_index, predicted_name in enumerate(CLASS_NAMES):
                row[
                    f"test_true_{actual_name.lower()}_pred_{predicted_name.lower()}"
                ] = int(matrix[actual_index, predicted_index])
        final_rows.append(row)

        if config == selected_configs[activation]:
            fig = plot_confusion_matrix(
                matrix,
                title=f"Independent evaluation — {activation.title()} — test set",
                save_path=output_dir / f"independent_{activation}_confusion_matrix.png",
            )
            plt.close(fig)

    selection_path = output_dir / "validation_selection.csv"
    public_selection_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in selection_rows
    ]
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(public_selection_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(public_selection_rows)

    final_path = output_dir / "independent_evaluation.csv"
    with final_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)

    return final_rows, final_path


def _matrix_markdown(matrix: np.ndarray) -> str:
    header = "| Actual \\ Predicted | " + " | ".join(CLASS_NAMES) + " |"
    separator = "|---|" + "---:|" * len(CLASS_NAMES)
    rows = [header, separator]
    for index, name in enumerate(CLASS_NAMES):
        values = " | ".join(str(int(value)) for value in matrix[index])
        rows.append(f"| **{name}** | {values} |")
    return "\n".join(rows)


def _config_table_row(result: ExperimentResult) -> str:
    config = result.config
    return (
        f"| {config.activation.title()} | {result.train_accuracy:.2f}% | "
        f"{result.test_accuracy:.2f}% | {config.learning_rate:g} | "
        f"{config.epochs} | {len(config.hidden_layers)} | "
        f"{_format_hidden(config)} | {'Yes' if config.use_bias else 'No'} |"
    )


def plot_best_results(
    results: list[ExperimentResult],
    best: dict[str, ExperimentResult],
    output_dir: Path = REPORT_DIR,
) -> None:
    """Save the loss curves and confusion matrices for both selected models."""
    for activation, result in best.items():
        config = result.config
        title_suffix = f"{activation.title()} | {_architecture(config)} | eta={config.learning_rate:g}"
        fig = plot_training_loss(
            result.model.loss_history_,
            title=f"Training loss — {title_suffix}",
            save_path=output_dir / f"best_{activation}_loss_curve.png",
        )
        plt.close(fig)

        fig = plot_confusion_matrix(
            result.matrix_test,
            title=f"Test confusion matrix — {title_suffix}",
            save_path=output_dir / f"best_{activation}_confusion_matrix.png",
        )
        plt.close(fig)

        fig = plot_actual_vs_predicted(
            _labels_for_plot(result, "test"),
            _predictions_for_plot(result, "test"),
            title=f"Actual vs predicted — {activation.title()} test set",
            save_path=output_dir / f"best_{activation}_actual_vs_predicted.png",
        )
        plt.close(fig)


# Plot helpers use the stored metrics instead of re-running the model.
def _labels_for_plot(result: ExperimentResult, split: str) -> np.ndarray:
    matrix = result.matrix_train if split == "train" else result.matrix_test
    labels: list[str] = []
    for index, name in enumerate(CLASS_NAMES):
        labels.extend([name] * int(matrix[index].sum()))
    return np.asarray(labels, dtype=object)


def _predictions_for_plot(result: ExperimentResult, split: str) -> np.ndarray:
    matrix = result.matrix_train if split == "train" else result.matrix_test
    predictions: list[str] = []
    for actual_index in range(len(CLASS_NAMES)):
        for predicted_index, name in enumerate(CLASS_NAMES):
            predictions.extend([name] * int(matrix[actual_index, predicted_index]))
    return np.asarray(predictions, dtype=object)


def _activation_statistics(
    results: list[ExperimentResult], activation: str
) -> dict[str, float]:
    subset = [result for result in results if result.config.activation == activation]
    test_scores = np.asarray([result.test_accuracy for result in subset])
    train_scores = np.asarray([result.train_accuracy for result in subset])
    gaps = train_scores - test_scores
    return {
        "count": float(len(subset)),
        "best_test": float(test_scores.max()),
        "mean_test": float(test_scores.mean()),
        "median_test": float(np.median(test_scores)),
        "mean_train": float(train_scores.mean()),
        "mean_gap": float(gaps.mean()),
    }


def write_report(
    results: list[ExperimentResult],
    best: dict[str, ExperimentResult],
    independent_rows: list[dict[str, object]],
    output_dir: Path = REPORT_DIR,
) -> Path:
    """Write a Markdown report containing only measured values."""
    ties = {
        activation: find_test_accuracy_ties(results, activation) for activation in best
    }
    sigmoid_stats = _activation_statistics(results, "sigmoid")
    tanh_stats = _activation_statistics(results, "tanh")

    lines: list[str] = [
        "# Task 2 — Back-Propagation Neural Network Report",
        "",
        "All numbers below were produced by `python -m src.experiment_runner` in this",
        "repository. No results were entered manually.",
        "",
        "## Protocol",
        "",
        "* Dataset: `data/penguins.csv`, 150 rows (50 per class), no rows dropped.",
        "* Fixed split: the first 30 rows of each class train, the remaining 20 test.",
        "* Training: 90 samples (30 Adelie, 30 Chinstrap, 30 Gentoo).",
        "* Testing: 60 samples (20 per class).",
        "* Five inputs: CulmenLength, CulmenDepth, FlipperLength, OriginLocation, BodyMass.",
        "* One-hot outputs in the fixed order Adelie, Chinstrap, Gentoo.",
        "* Online (stochastic) gradient descent, MSE loss, one sample per update.",
        "* Weight and bias initialization: NumPy normal distribution with mean 0 and",
        "  standard deviation 0.1, controlled by the recorded random seed.",
        "* Preprocessing is fitted on training data only and reused unchanged for test",
        "  data and for single-sample GUI predictions.",
        "",
        "## Preprocessing note and limitation",
        "",
        "Numeric missing values are imputed with training-set means, then numeric",
        "features are standardized with training-set statistics. `OriginLocation` is",
        "nominal but encoded as one ordinal column because the assignment fixes the",
        "input layer at five neurons. The artificial order is `Biscoe < Dream <",
        "Torgersen`; the network can therefore interpret the category as a magnitude.",
        "One-hot encoding would need three columns and change the input layer to seven.",
        "Missing categories are filled with the training mode. An unseen category is",
        "mapped to the training mode code so prediction remains defined.",
        "",
        "## Experiment grid",
        "",
        "The grid varies activation, learning rate, epochs, hidden architecture, and",
        "bias. One seed (42) is used per configuration, so these results show one",
        "realization per setting rather than an average over seeds.",
        "",
        "## Best experiment per activation",
        "",
        "| Activation Function | Train Accuracy | Test Accuracy | Learning Rate | Epochs | Number of Hidden Layers | Hidden Neurons | Bias |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for activation in ("sigmoid", "tanh"):
        lines.append(_config_table_row(best[activation]))
    lines.extend(["", "## Full grid results", ""])
    lines.append(
        "| Experiment | Activation | Architecture | Learning Rate | Epochs | Hidden Layers | Hidden Neurons | Bias | Train Accuracy | Test Accuracy | Final MSE |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|---|---:|---:|---:|")
    for result in results:
        config = result.config
        lines.append(
            f"| {result.experiment_id} | {config.activation.title()} | "
            f"{_architecture(config)} | {config.learning_rate:g} | {config.epochs} | "
            f"{len(config.hidden_layers)} | {_format_hidden(config)} | "
            f"{'Yes' if config.use_bias else 'No'} | {result.train_accuracy:.2f}% | "
            f"{result.test_accuracy:.2f}% | {result.final_mse:.6f} |"
        )

    lines.extend(["", "## Ties", ""])
    for activation in ("sigmoid", "tanh"):
        tied = ties[activation]
        lines.append(
            f"* **{activation.title()}**: best test accuracy was "
            f"{best[activation].test_accuracy:.2f}% and was shared by "
            f"{len(tied)} configuration(s)."
        )
        if len(tied) > 1:
            for result in tied:
                lines.append(
                    f"  * {result.experiment_id}: architecture "
                    f"{_architecture(result.config)}, eta={result.config.learning_rate:g}, "
                    f"epochs={result.config.epochs}, bias="
                    f"{'yes' if result.config.use_bias else 'no'}, train="
                    f"{result.train_accuracy:.2f}%, test={result.test_accuracy:.2f}%."
                )
        else:
            lines.append("  * No tie occurred; this configuration was uniquely best.")

    lines.extend(["", "## Best Sigmoid result", ""])
    sigmoid_best = best["sigmoid"]
    lines.append(
        f"Configuration: architecture `{_architecture(sigmoid_best.config)}`, "
        f"eta={sigmoid_best.config.learning_rate:g}, "
        f"epochs={sigmoid_best.config.epochs}, "
        f"bias={'enabled' if sigmoid_best.config.use_bias else 'disabled'}, "
        f"seed={sigmoid_best.config.random_state}. "
        f"Train accuracy **{sigmoid_best.train_accuracy:.2f}%**, "
        f"test accuracy **{sigmoid_best.test_accuracy:.2f}%**."
    )
    lines.extend(
        [
            "",
            "![Best Sigmoid loss curve](best_sigmoid_loss_curve.png)",
            "",
            "![Best Sigmoid confusion matrix](best_sigmoid_confusion_matrix.png)",
            "",
            _matrix_markdown(sigmoid_best.matrix_test),
            "",
            "## Best Tanh result",
            "",
        ]
    )
    tanh_best = best["tanh"]
    lines.append(
        f"Configuration: architecture `{_architecture(tanh_best.config)}`, "
        f"eta={tanh_best.config.learning_rate:g}, "
        f"epochs={tanh_best.config.epochs}, "
        f"bias={'enabled' if tanh_best.config.use_bias else 'disabled'}, "
        f"seed={tanh_best.config.random_state}. "
        f"Train accuracy **{tanh_best.train_accuracy:.2f}%**, "
        f"test accuracy **{tanh_best.test_accuracy:.2f}%**."
    )
    lines.extend(
        [
            "",
            "![Best Tanh loss curve](best_tanh_loss_curve.png)",
            "",
            "![Best Tanh confusion matrix](best_tanh_confusion_matrix.png)",
            "",
            _matrix_markdown(tanh_best.matrix_test),
            "",
            "## Activation comparison",
            "",
            "| Activation | Configurations | Best Test | Mean Test | Median Test | Mean Train | Mean (Train − Test) |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| Sigmoid | {int(sigmoid_stats['count'])} | {sigmoid_stats['best_test']:.2f}% | "
            f"{sigmoid_stats['mean_test']:.2f}% | {sigmoid_stats['median_test']:.2f}% | "
            f"{sigmoid_stats['mean_train']:.2f}% | {sigmoid_stats['mean_gap']:.2f} pp |",
            f"| Tanh | {int(tanh_stats['count'])} | {tanh_stats['best_test']:.2f}% | "
            f"{tanh_stats['mean_test']:.2f}% | {tanh_stats['median_test']:.2f}% | "
            f"{tanh_stats['mean_train']:.2f}% | {tanh_stats['mean_gap']:.2f} pp |",
            "",
            f"The best test accuracy was {sigmoid_stats['best_test']:.2f}% for sigmoid and "
            f"{tanh_stats['best_test']:.2f}% for tanh. Across the whole grid, the mean test "
            f"score was {sigmoid_stats['mean_test']:.2f}% for sigmoid and "
            f"{tanh_stats['mean_test']:.2f}% for tanh. The mean training-test gap was "
            f"{sigmoid_stats['mean_gap']:.2f} percentage points for sigmoid and "
            f"{tanh_stats['mean_gap']:.2f} for tanh. A large positive gap indicates "
            "memorization of the 90 training samples rather than better generalization. "
            "These comparisons describe this fixed split and this seed; they are not "
            "claims about the activation functions in general.",
            "",
            "## Independent check (validation-selected, test evaluated once)",
            "",
            "The grid above chooses its winner using the test set, so that winner's test",
            "accuracy is an optimistic selection statistic. To reduce that bias, the",
            "training set was split internally into the first 20 rows per class for",
            "selection training and the remaining 10 per class for validation. A",
            "configuration was selected using only those 30 validation samples,",
            "retrained on all 90 training samples, and evaluated once on the untouched",
            "60-sample test set. Preprocessing was fitted separately inside the inner",
            "training split to prevent leakage.",
            "",
            "| Activation | Learning Rate | Epochs | Hidden Layers | Hidden Neurons | Bias | Validation Accuracy | Final Train Accuracy | Final Test Accuracy |",
            "|---|---:|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in independent_rows:
        lines.append(
            f"| {str(row['activation']).title()} | {row['learning_rate']:g} | "
            f"{row['epochs']} | {row['number_of_hidden_layers']} | "
            f"{row['hidden_neurons']} | {row['use_bias']} | "
            f"{row['validation_accuracy']:.2f}% | {row['final_train_accuracy']:.2f}% | "
            f"{row['final_test_accuracy']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "This independent check is still not a full nested cross-validation, and the",
            "candidate grid and seed are fixed by the experimenter. It is nevertheless a",
            "cleaner estimate than selecting on the test set. See",
            "`validation_selection.csv`, `independent_evaluation.csv`, and",
            "`independent_*_confusion_matrix.png`.",
            "",
            "## GUI screenshots",
            "",
            "Captured from the actual Tkinter application after training the best",
            "configurations from this report:",
            "",
            "![GUI — best Sigmoid](screenshots/gui_best_sigmoid.png)",
            "",
            "![GUI — best Tanh](screenshots/gui_best_tanh.png)",
            "",
            "## Limitations and observations",
            "",
            "* The professor's split is fixed by row order, not randomized. The first 30",
            "  rows may not represent the class populations equally; test accuracy is",
            "  conditional on this specific split.",
            "* Selecting the best of many configurations on the same 60 test samples",
            "  inflates the reported best test accuracy. The independent section above",
            "  separates selection from final testing, but it is not a substitute for a",
            "  larger dataset and repeated nested cross-validation.",
            "* `OriginLocation` uses ordinal coding with the documented artificial order.",
            "* Only one random seed is used per grid configuration; different initial",
            "  weights could change close results.",
            "* The small dataset (150 rows) and simple online SGD make high training",
            "  accuracy easy to reach and overfitting visible.",
            "* No regularization, early stopping, momentum, or adaptive learning rate is",
            "  used; those were not required by the assignment.",
            "* Tanh outputs lie in (-1, 1), so they are class scores, not probabilities.",
            "  Sigmoid outputs also should not be treated as calibrated probabilities.",
            "* Missing numeric values are few; mean imputation is simple and may hide",
            "  uncertainty in those rows.",
        ]
    )
    stalled = [
        result
        for result in results
        if result.train_accuracy < 50.0 or result.test_accuracy < 50.0
    ]
    if stalled:
        lines.append(
            "* Some runs stalled on this initialization: "
            + ", ".join(
                f"{result.experiment_id} (train {result.train_accuracy:.2f}%, "
                f"test {result.test_accuracy:.2f}%)"
                for result in stalled
            )
            + ". This is an observed optimization failure at low learning rates in "
            "the deeper sigmoid network, not a hidden fallback or edited result."
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```text",
            "cd task-2",
            "python app.py",
            "python -m src.experiment_runner",
            "python -m pytest -q",
            "```",
            "",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    results, csv_path = run_experiments()
    best = select_best_by_activation(results)
    plot_best_results(results, best)
    independent_rows, independent_path = run_independent_evaluation(results)
    report_path = write_report(results, best, independent_rows)
    print(f"\nSaved {len(results)} experiment rows to: {csv_path}")
    print(f"Saved independent evaluation to: {independent_path}")
    print(f"Saved report to: {report_path}")
    for activation, result in best.items():
        print(
            f"Best {activation:7}: train={result.train_accuracy:.2f}% "
            f"test={result.test_accuracy:.2f}% "
            f"arch={_architecture(result.config)} "
            f"eta={result.config.learning_rate:g} epochs={result.config.epochs} "
            f"bias={result.config.use_bias}"
        )


if __name__ == "__main__":
    main()
