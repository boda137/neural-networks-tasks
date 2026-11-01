"""Run a small reproducible experiment set and save CSV results/plots."""

from pathlib import Path
import csv

from src.data_loader import load_penguins
from src.data_splitter import split_two_classes
from src.preprocessing import FeaturePreprocessor
from src.perceptron import Perceptron
from src.adaline import Adaline
from src.evaluation import accuracy_manual, confusion_matrix_manual
from src.visualization import plot_decision_boundary

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "penguins.csv"
REPORT_DIR = Path(__file__).resolve().parents[1] / "report"


def run_experiments(output_dir: Path = REPORT_DIR):
    """Run five feature/class combinations per algorithm and save outputs."""
    data = load_penguins(DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = [
        ("Adelie", "Gentoo", ("CulmenLength", "FlipperLength")),
        ("Adelie", "Chinstrap", ("CulmenLength", "CulmenDepth")),
        ("Chinstrap", "Gentoo", ("CulmenDepth", "BodyMass")),
        ("Adelie", "Gentoo", ("CulmenDepth", "BodyMass")),
        ("Adelie", "Chinstrap", ("FlipperLength", "BodyMass")),
    ]
    rows = []
    for algorithm in ("Perceptron", "ADALINE"):
        for index, (class_1, class_2, features) in enumerate(experiments, start=1):
            train, test = split_two_classes(data, class_1, class_2, random_state=40 + index)
            prep = FeaturePreprocessor(features).fit(train)
            X_train, X_test = prep.transform(train), prep.transform(test)
            y_train = train["Species"].map({class_1: -1, class_2: 1}).to_numpy()
            y_test = test["Species"].map({class_1: -1, class_2: 1}).to_numpy()
            if algorithm == "Perceptron":
                model = Perceptron(learning_rate=0.01, epochs=100, use_bias=True,
                                   random_state=100 + index).fit(X_train, y_train)
                training_detail = f"last_epoch_errors={model.errors_[-1]}"
            else:
                model = Adaline(learning_rate=0.01, epochs=100, mse_threshold=0.01,
                                use_bias=True, random_state=100 + index).fit(X_train, y_train)
                training_detail = f"final_mse={model.final_mse_:.5f}"
            predictions = model.predict(X_test)
            matrix = confusion_matrix_manual(y_test, predictions)
            accuracy = accuracy_manual(y_test, predictions)
            slug = f"{algorithm.lower()}_experiment_{index}"
            fig = plot_decision_boundary(
                X_train, y_train, X_test, y_test, model, features,
                (class_1, class_2), algorithm, output_dir / f"{slug}.png"
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
            rows.append({
                "algorithm": algorithm, "experiment": index,
                "class_1": class_1, "class_2": class_2,
                "feature_1": features[0], "feature_2": features[1],
                "accuracy": round(accuracy, 4),
                "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
                "training_detail": training_detail,
            })
    csv_path = output_dir / "experiment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows, csv_path


if __name__ == "__main__":
    results, path = run_experiments()
    print(f"Saved {len(results)} experiment results to: {path}")
    for row in results:
        print(f"{row['algorithm']:10} | {row['class_1']:10} vs {row['class_2']:10} | "
              f"{row['feature_1']}, {row['feature_2']} | accuracy={row['accuracy']:.1%} | "
              f"{row['training_detail']}")
