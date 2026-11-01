"""Tkinter GUI for the  Perceptron/ADALINE task."""

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.adaline import Adaline
from src.data_loader import NUMERIC_FEATURES, load_penguins
from src.data_splitter import split_two_classes
from src.evaluation import accuracy_manual, confusion_matrix_manual
from src.perceptron import Perceptron
from src.preprocessing import FeaturePreprocessor
from src.visualization import plot_decision_boundary

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "penguins.csv"
PLOT_PATH = ROOT / "report" / "latest_result.png"
CLASSES = ["Adelie", "Chinstrap", "Gentoo"]
FEATURES = NUMERIC_FEATURES + ["OriginLocation"]


class ClassifierApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Lab 3 — Perceptron and ADALINE")
        self.root.geometry("1120x760")
        self.data = load_penguins(DATA_PATH)
        self._build_ui()

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(side=tk.TOP, fill=tk.X)

        self.feature_1 = tk.StringVar(value="CulmenLength")
        self.feature_2 = tk.StringVar(value="FlipperLength")
        self.class_1 = tk.StringVar(value="Adelie")
        self.class_2 = tk.StringVar(value="Gentoo")
        self.algorithm = tk.StringVar(value="Perceptron")
        self.learning_rate = tk.StringVar(value="0.01")
        self.epochs = tk.StringVar(value="100")
        self.mse_threshold = tk.StringVar(value="0.01")
        self.use_bias = tk.BooleanVar(value=True)

        def add_combo(label, variable, values, row, column):
            ttk.Label(controls, text=label).grid(
                row=row, column=column, sticky="w", padx=5, pady=4
            )
            ttk.Combobox(
                controls,
                textvariable=variable,
                values=values,
                state="readonly",
                width=18,
            ).grid(row=row, column=column + 1, sticky="ew", padx=5, pady=4)

        add_combo("Feature 1", self.feature_1, FEATURES, 0, 0)
        add_combo("Feature 2", self.feature_2, FEATURES, 0, 2)
        add_combo("Class 1", self.class_1, CLASSES, 1, 0)
        add_combo("Class 2", self.class_2, CLASSES, 1, 2)
        add_combo("Algorithm", self.algorithm, ["Perceptron", "ADALINE"], 2, 0)

        ttk.Label(controls, text="Learning rate (η)").grid(
            row=2, column=2, sticky="w", padx=5, pady=4
        )
        ttk.Entry(controls, textvariable=self.learning_rate, width=12).grid(
            row=2, column=3, sticky="w", padx=5, pady=4
        )
        ttk.Label(controls, text="Epochs").grid(
            row=3, column=0, sticky="w", padx=5, pady=4
        )
        ttk.Entry(controls, textvariable=self.epochs, width=12).grid(
            row=3, column=1, sticky="w", padx=5, pady=4
        )
        ttk.Label(controls, text="MSE threshold (ADALINE)").grid(
            row=3, column=2, sticky="w", padx=5, pady=4
        )
        ttk.Entry(controls, textvariable=self.mse_threshold, width=12).grid(
            row=3, column=3, sticky="w", padx=5, pady=4
        )
        ttk.Checkbutton(controls, text="Use Bias", variable=self.use_bias).grid(
            row=4, column=0, sticky="w", padx=5, pady=4
        )
        ttk.Button(controls, text="Train and Evaluate", command=self.train).grid(
            row=4, column=2, columnspan=2, sticky="ew", padx=5, pady=6
        )

        self.result_text = tk.Text(
            self.root, height=8, wrap="word", font=("Consolas", 10)
        )
        self.result_text.pack(side=tk.TOP, fill=tk.X, padx=12, pady=6)
        self.plot_frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def train(self):
        try:
            f1, f2 = self.feature_1.get(), self.feature_2.get()
            c1, c2 = self.class_1.get(), self.class_2.get()
            if f1 == f2:
                raise ValueError("Choose two different features.")
            if c1 == c2:
                raise ValueError("Choose two different classes.")
            lr = float(self.learning_rate.get())
            epochs = int(self.epochs.get())
            threshold = float(self.mse_threshold.get())
            if lr <= 0 or epochs < 1 or threshold < 0:
                raise ValueError(
                    "Learning rate must be > 0, epochs >= 1, and MSE threshold >= 0."
                )

            train, test = split_two_classes(self.data, c1, c2, random_state=42)
            preprocessor = FeaturePreprocessor((f1, f2)).fit(train)
            X_train = preprocessor.transform(train)
            X_test = preprocessor.transform(test)
            label_map = {c1: -1, c2: 1}
            y_train = train["Species"].map(label_map).to_numpy()
            y_test = test["Species"].map(label_map).to_numpy()

            if self.algorithm.get() == "Perceptron":
                model = Perceptron(
                    learning_rate=lr,
                    epochs=epochs,
                    use_bias=self.use_bias.get(),
                    random_state=42,
                ).fit(X_train, y_train)
                training_summary = f"Epochs used: {len(model.errors_)}\nTraining errors (last epoch): {model.errors_[-1]}"
            else:
                model = Adaline(
                    learning_rate=lr,
                    epochs=epochs,
                    mse_threshold=threshold,
                    use_bias=self.use_bias.get(),
                    random_state=42,
                ).fit(X_train, y_train)
                training_summary = f"Epochs used: {len(model.mse_history_)}\nFinal training MSE: {model.final_mse_:.5f}"

            predictions = model.predict(X_test)
            matrix = confusion_matrix_manual(y_test, predictions, labels=(-1, 1))
            accuracy = accuracy_manual(y_test, predictions)
            fig = plot_decision_boundary(
                X_train,
                y_train,
                X_test,
                y_test,
                model,
                (f1, f2),
                (c1, c2),
                self.algorithm.get(),
                PLOT_PATH,
            )
            for child in self.plot_frame.winfo_children():
                child.destroy()
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            plt.close(fig)

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(
                tk.END,
                f"Algorithm: {self.algorithm.get()} | Features: {f1}, {f2}\n"
                f"Classes: {c1} (-1) vs {c2} (+1) | Train: 30/class | Test: 20/class\n"
                f"Test accuracy: {accuracy:.2%}\n"
                f"Confusion matrix (rows=true [-1,+1], columns=predicted [-1,+1]):\n"
                f"[[{matrix[0, 0]:2d}, {matrix[0, 1]:2d}], [{matrix[1, 0]:2d}, {matrix[1, 1]:2d}]]\n"
                f"{training_summary}\nPlot saved to: {PLOT_PATH.relative_to(ROOT)}",
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Could not run experiment", str(exc))


def main():
    root = tk.Tk()
    ClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
