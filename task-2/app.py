"""Tkinter GUI for Task 2: back-propagation on a multi-layer network."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.data_loader import NUMERIC_FEATURES, load_penguins
from src.data_splitter import split_train_test_per_class
from src.encoding import (
    CATEGORICAL_FEATURE,
    CLASS_NAMES,
    FEATURE_NAMES,
    decode_labels,
    encode_labels,
)
from src.evaluation import evaluate_predictions
from src.neural_network import BackPropagationNetwork
from src.preprocessing import PenguinPreprocessor
from src.visualization import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_training_loss,
)

TASK_ROOT = Path(__file__).resolve().parent
DATA_PATH = TASK_ROOT / "data" / "penguins.csv"
REPORT_DIR = TASK_ROOT / "report"
MAX_HIDDEN_LAYERS = 6


class BackPropagationApp:
    """Graphical interface for configuring, training, and testing the network."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Task 2 — Back-Propagation Multi-Layer Neural Network")
        self.root.geometry("1220x800")
        self.root.minsize(1040, 720)

        self.data = load_penguins(DATA_PATH)
        self.train_data, self.test_data = split_train_test_per_class(self.data)
        self.preprocessor: PenguinPreprocessor | None = None
        self.model: BackPropagationNetwork | None = None

        self._build_ui()
        self._refresh_hidden_entries()
        self._set_sample_defaults()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.LabelFrame(main, text="Network configuration", padding=8)
        config_frame.pack(fill=tk.X, pady=(0, 8))

        self.hidden_count = tk.StringVar(value="2")
        self.layer_vars = [tk.StringVar(value="8") for _ in range(MAX_HIDDEN_LAYERS)]
        self.learning_rate = tk.StringVar(value="0.05")
        self.epochs = tk.StringVar(value="1000")
        self.random_seed = tk.StringVar(value="42")
        self.activation = tk.StringVar(value="Sigmoid")
        self.use_bias = tk.BooleanVar(value=True)

        ttk.Label(config_frame, text="Number of hidden layers").grid(
            row=0, column=0, sticky="w", padx=5, pady=4
        )
        hidden_selector = ttk.Combobox(
            config_frame,
            textvariable=self.hidden_count,
            values=[str(value) for value in range(1, MAX_HIDDEN_LAYERS + 1)],
            state="readonly",
            width=4,
        )
        hidden_selector.grid(row=0, column=1, sticky="w", padx=5, pady=4)
        hidden_selector.bind("<<ComboboxSelected>>", self._on_hidden_count_changed)

        self.layers_frame = ttk.Frame(config_frame)
        self.layers_frame.grid(
            row=0, column=2, columnspan=8, sticky="w", padx=5, pady=4
        )
        self.layer_labels: list[ttk.Label] = []
        self.layer_entries: list[ttk.Entry] = []
        for index in range(MAX_HIDDEN_LAYERS):
            label = ttk.Label(self.layers_frame, text=f"H{index + 1}")
            entry = ttk.Entry(
                self.layers_frame,
                textvariable=self.layer_vars[index],
                width=5,
            )
            label.grid(row=0, column=2 * index, padx=(4, 1), pady=2)
            entry.grid(row=0, column=2 * index + 1, padx=(1, 8), pady=2)
            self.layer_labels.append(label)
            self.layer_entries.append(entry)

        ttk.Label(config_frame, text="Learning rate (η)").grid(
            row=1, column=0, sticky="w", padx=5, pady=4
        )
        ttk.Entry(config_frame, textvariable=self.learning_rate, width=10).grid(
            row=1, column=1, sticky="w", padx=5, pady=4
        )

        ttk.Label(config_frame, text="Epochs").grid(
            row=1, column=2, sticky="w", padx=5, pady=4
        )
        ttk.Entry(config_frame, textvariable=self.epochs, width=10).grid(
            row=1, column=3, sticky="w", padx=5, pady=4
        )

        ttk.Label(config_frame, text="Random seed").grid(
            row=1, column=4, sticky="w", padx=5, pady=4
        )
        ttk.Entry(config_frame, textvariable=self.random_seed, width=8).grid(
            row=1, column=5, sticky="w", padx=5, pady=4
        )

        ttk.Label(config_frame, text="Activation").grid(
            row=1, column=6, sticky="w", padx=5, pady=4
        )
        ttk.Combobox(
            config_frame,
            textvariable=self.activation,
            values=["Sigmoid", "Tanh"],
            state="readonly",
            width=10,
        ).grid(row=1, column=7, sticky="w", padx=5, pady=4)

        ttk.Checkbutton(config_frame, text="Use bias", variable=self.use_bias).grid(
            row=1, column=8, sticky="w", padx=5, pady=4
        )

        self.train_button = ttk.Button(
            config_frame,
            text="Train and Evaluate",
            command=self.train_and_evaluate,
        )
        self.train_button.grid(row=1, column=9, sticky="ew", padx=5, pady=4)

        sample_frame = ttk.LabelFrame(
            main, text="Classify one new penguin sample", padding=8
        )
        sample_frame.pack(fill=tk.X, pady=(0, 8))

        self.sample_vars: dict[str, tk.StringVar] = {
            feature: tk.StringVar() for feature in NUMERIC_FEATURES
        }
        self.origin_var = tk.StringVar(value="Dream")

        for column_index, feature in enumerate(NUMERIC_FEATURES):
            ttk.Label(sample_frame, text=feature).grid(
                row=0, column=2 * column_index, sticky="w", padx=(5, 2), pady=3
            )
            ttk.Entry(
                sample_frame,
                textvariable=self.sample_vars[feature],
                width=10,
            ).grid(
                row=0,
                column=2 * column_index + 1,
                sticky="w",
                padx=(2, 8),
                pady=3,
            )

        ttk.Label(sample_frame, text=CATEGORICAL_FEATURE).grid(
            row=0, column=8, sticky="w", padx=(5, 2), pady=3
        )
        self.origin_values = sorted(
            str(value) for value in self.data[CATEGORICAL_FEATURE].dropna().unique()
        )
        ttk.Combobox(
            sample_frame,
            textvariable=self.origin_var,
            values=self.origin_values,
            state="readonly",
            width=10,
        ).grid(row=0, column=9, sticky="w", padx=(2, 8), pady=3)

        ttk.Button(
            sample_frame,
            text="Classify Sample",
            command=self.classify_sample,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=(6, 2))

        self.prediction_var = tk.StringVar(value="Train a model first.")
        ttk.Label(
            sample_frame,
            textvariable=self.prediction_var,
            foreground="#0b4f6c",
            wraplength=980,
        ).grid(
            row=1,
            column=2,
            columnspan=8,
            sticky="w",
            padx=5,
            pady=(6, 2),
        )

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.loss_tab = ttk.Frame(self.notebook)
        self.confusion_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.loss_tab, text="Training loss")
        self.notebook.add(self.confusion_tab, text="Confusion matrix")
        self.notebook.add(self.comparison_tab, text="Actual vs predicted")
        self._show_placeholder(
            self.loss_tab, "Train the network to see the loss curve."
        )
        self._show_placeholder(
            self.confusion_tab, "Train the network to see the confusion matrix."
        )
        self._show_placeholder(
            self.comparison_tab, "Train the network to compare actual and predicted."
        )

        result_frame = ttk.LabelFrame(main, text="Latest run", padding=4)
        result_frame.pack(fill=tk.X)
        self.result_text = tk.Text(
            result_frame, height=10, wrap="word", font=("Consolas", 10)
        )
        self.result_text.pack(fill=tk.X)

    def _on_hidden_count_changed(self, _event=None) -> None:
        self._refresh_hidden_entries()

    def _refresh_hidden_entries(self) -> None:
        try:
            count = int(self.hidden_count.get())
        except ValueError:
            count = 1
        for index, (label, entry) in enumerate(
            zip(self.layer_labels, self.layer_entries)
        ):
            if index < count:
                label.grid()
                entry.grid()
            else:
                label.grid_remove()
                entry.grid_remove()

    def _set_sample_defaults(self) -> None:
        medians = self.data[list(NUMERIC_FEATURES)].median(numeric_only=True)
        for feature in NUMERIC_FEATURES:
            value = float(medians[feature])
            text = f"{value:.0f}" if feature == "BodyMass" else f"{value:.1f}"
            self.sample_vars[feature].set(text)
        most_common = self.data[CATEGORICAL_FEATURE].mode(dropna=True)
        if not most_common.empty:
            self.origin_var.set(str(most_common.iloc[0]))

    @staticmethod
    def _show_placeholder(parent: ttk.Frame, message: str) -> None:
        for child in parent.winfo_children():
            child.destroy()
        ttk.Label(parent, text=message, anchor="center").pack(
            fill=tk.BOTH, expand=True, padx=20, pady=20
        )

    @staticmethod
    def _embed_figure(parent: ttk.Frame, fig: plt.Figure) -> None:
        for child in parent.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    # -------------------------------------------------------------- actions
    def _read_configuration(self) -> dict[str, object]:
        try:
            hidden_count = int(self.hidden_count.get())
        except ValueError as exc:
            raise ValueError("Choose a valid number of hidden layers.") from exc
        if hidden_count < 1 or hidden_count > MAX_HIDDEN_LAYERS:
            raise ValueError(
                f"Number of hidden layers must be between 1 and {MAX_HIDDEN_LAYERS}."
            )

        hidden_layers: list[int] = []
        for index in range(hidden_count):
            raw = self.layer_vars[index].get().strip()
            if not raw:
                raise ValueError(f"Enter a neuron count for hidden layer H{index + 1}.")
            try:
                width = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Neurons in hidden layer H{index + 1} must be a whole number."
                ) from exc
            if width <= 0 or width > 512:
                raise ValueError(
                    f"Neurons in hidden layer H{index + 1} must be between 1 and 512."
                )
            hidden_layers.append(width)

        try:
            learning_rate = float(self.learning_rate.get())
        except ValueError as exc:
            raise ValueError("Learning rate must be a number.") from exc
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be a finite value greater than 0.")

        try:
            epochs = int(self.epochs.get())
        except ValueError as exc:
            raise ValueError("Epochs must be a whole number.") from exc
        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0.")

        try:
            random_seed = int(self.random_seed.get())
        except ValueError as exc:
            raise ValueError("Random seed must be a whole number.") from exc
        if random_seed < 0:
            raise ValueError("Random seed must be 0 or greater.")

        return {
            "hidden_layers": hidden_layers,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "random_seed": random_seed,
            "activation": self.activation.get(),
            "use_bias": bool(self.use_bias.get()),
        }

    def train_and_evaluate(self) -> None:
        try:
            config = self._read_configuration()
        except ValueError as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return

        self.train_button.configure(state=tk.DISABLED)
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            preprocessor = PenguinPreprocessor().fit(self.train_data)
            X_train = preprocessor.transform(self.train_data)
            X_test = preprocessor.transform(self.test_data)
            y_train = encode_labels(self.train_data["Species"].to_numpy())
            encode_labels(self.test_data["Species"].to_numpy())

            model = BackPropagationNetwork(
                hidden_layers=config["hidden_layers"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                use_bias=config["use_bias"],
                activation=config["activation"],
                random_state=config["random_seed"],
            ).fit(X_train, y_train)

            train_predictions = np.asarray(
                decode_labels(model.predict(X_train)), dtype=object
            )
            test_predictions = np.asarray(
                decode_labels(model.predict(X_test)), dtype=object
            )
            train_metrics = evaluate_predictions(
                self.train_data["Species"].to_numpy(), train_predictions
            )
            test_metrics = evaluate_predictions(
                self.test_data["Species"].to_numpy(), test_predictions
            )

            self.preprocessor = preprocessor
            self.model = model
            self._update_outputs(
                model, config, train_metrics, test_metrics, test_predictions
            )
            self._update_plots(model, test_metrics, test_predictions, config)
            self.prediction_var.set(
                "Model trained. Edit the sample fields and classify."
            )
        except Exception as exc:  # noqa: BLE001 - GUI must report all failures
            messagebox.showerror("Training failed", str(exc))
        finally:
            self.root.configure(cursor="")
            self.train_button.configure(state=tk.NORMAL)

    def _update_outputs(
        self,
        model: BackPropagationNetwork,
        config: dict[str, object],
        train_metrics: dict[str, object],
        test_metrics: dict[str, object],
        test_predictions: np.ndarray,
    ) -> None:
        matrix = np.asarray(test_metrics["confusion_matrix"])
        train_accuracy = float(train_metrics["accuracy"]) * 100.0
        test_accuracy = float(test_metrics["accuracy"]) * 100.0
        bias_text = "enabled" if config["use_bias"] else "disabled"
        lines = [
            (
                f"Architecture: {model.architecture_description}    "
                f"Activation: {model.activation_name.title()}    Bias: {bias_text}"
            ),
            (
                f"Learning rate: {config['learning_rate']:g}    "
                f"Epochs: {config['epochs']}    Seed: {config['random_seed']}"
            ),
            (
                f"Training samples: {len(self.train_data)} (30 per class)    "
                f"Testing samples: {len(self.test_data)} (20 per class)"
            ),
            f"Final training MSE: {model.loss_history_[-1]:.6f}",
            (
                f"Training accuracy: {train_accuracy:.2f}% "
                f"({train_metrics['correct']}/{train_metrics['total']})    "
                f"Testing accuracy: {test_accuracy:.2f}% "
                f"({test_metrics['correct']}/{test_metrics['total']})"
            ),
            "Confusion matrix (rows = actual, columns = predicted):",
            " " * 13 + "".join(f"{name:>12}" for name in CLASS_NAMES),
        ]
        for name, row in zip(CLASS_NAMES, matrix):
            lines.append(f"{name:<13}" + "".join(f"{int(value):>12}" for value in row))
        lines.append(
            "Plots are also saved as latest_loss_curve.png, "
            "latest_confusion_matrix.png, and latest_actual_vs_predicted.png."
        )
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(lines))

    def _update_plots(
        self,
        model: BackPropagationNetwork,
        test_metrics: dict[str, object],
        test_predictions: np.ndarray,
        config: dict[str, object],
    ) -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        title_suffix = (
            f"{model.activation_name.title()} | {model.architecture_description} | "
            f"eta={config['learning_rate']:g}"
        )
        loss_fig = plot_training_loss(
            model.loss_history_,
            title=f"Training loss — {title_suffix}",
            save_path=REPORT_DIR / "latest_loss_curve.png",
        )
        self._embed_figure(self.loss_tab, loss_fig)

        matrix = np.asarray(test_metrics["confusion_matrix"])
        confusion_fig = plot_confusion_matrix(
            matrix,
            title=f"Test confusion matrix — {title_suffix}",
            save_path=REPORT_DIR / "latest_confusion_matrix.png",
        )
        self._embed_figure(self.confusion_tab, confusion_fig)

        comparison_fig = plot_actual_vs_predicted(
            self.test_data["Species"].to_numpy(),
            test_predictions,
            title=f"Actual vs predicted — {title_suffix}",
            save_path=REPORT_DIR / "latest_actual_vs_predicted.png",
        )
        self._embed_figure(self.comparison_tab, comparison_fig)
        self.notebook.select(self.loss_tab)

    def classify_sample(self) -> None:
        if self.model is None or self.preprocessor is None:
            messagebox.showwarning(
                "No trained model", "Train the network before classifying a sample."
            )
            return

        try:
            values: dict[str, object] = {}
            for feature in NUMERIC_FEATURES:
                raw = self.sample_vars[feature].get().strip()
                if not raw:
                    raise ValueError(f"Enter a value for {feature}.")
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(f"{feature} must be a finite number.")
                values[feature] = value

            origin = self.origin_var.get().strip()
            if not origin:
                raise ValueError(f"Choose a value for {CATEGORICAL_FEATURE}.")
            values[CATEGORICAL_FEATURE] = origin

            sample = pd.DataFrame(
                [{feature: values[feature] for feature in FEATURE_NAMES}],
                columns=list(FEATURE_NAMES),
            )
            transformed = self.preprocessor.transform(sample)
            scores = self.model.predict_proba(transformed)[0]
            predicted_index = int(np.argmax(scores))
            predicted_name = decode_labels([predicted_index])[0]
            score_text = "    ".join(
                f"{name}={score:.4f}" for name, score in zip(CLASS_NAMES, scores)
            )
            self.prediction_var.set(
                f"Predicted species: {predicted_name}    "
                f"Output activations: {score_text}"
            )
        except Exception as exc:  # noqa: BLE001 - GUI must report all failures
            messagebox.showerror("Could not classify sample", str(exc))


def main() -> None:
    root = tk.Tk()
    BackPropagationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
