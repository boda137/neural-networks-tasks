# Neural Networks Tasks

University coursework for the **Neural Networks** course. Both assignments are implemented **from scratch** (NumPy-based, no TensorFlow / PyTorch / Keras / scikit-learn models) and ship with a Tkinter GUI.

| Task | Topic | Folder |
|---|---|---|
| 1 | Single-Layer Perceptron and ADALINE | [`task-1/`](task-1/) |
| 2 | Multi-Layer Neural Network trained with Back-Propagation | [`task-2/`](task-2/) |

## Repository structure

```text
Neural-Networks-Tasks/
├── README.md
├── .gitignore
├── task-1/          # Perceptron & ADALINE
└── task-2/          # Back-propagation network
    ├── app.py              # Tkinter GUI
    ├── src/                # network, preprocessing, evaluation, experiments
    ├── tests/              # pytest suite (40 tests)
    ├── data/penguins.csv
    ├── report/             # REPORT.md, CSV results, plots, GUI screenshots
    ├── requirements.txt
    └── README.md
```

## Requirements

* Python 3.10 or newer (developed with 3.13)
* Tk support for the GUIs (bundled with the python.org installers for Windows/macOS; on Debian/Ubuntu run `sudo apt install python3-tk`)

Each task has its own `requirements.txt`. Use a separate virtual environment per task, or one shared environment.

---

## Task 1 — Single-Layer Perceptron and ADALINE

Two binary linear classifiers written in NumPy, with a Tkinter GUI. The user selects two penguin species and two features; the app trains a **Perceptron** (online error-correction rule) or **ADALINE** (batch gradient descent on MSE), reports accuracy and the confusion matrix, and plots the decision boundary.

* 30 training / 20 test samples per class, train-only imputation and standardization
* Manual confusion matrix and accuracy (no scikit-learn metrics)
* 5 experiments per algorithm, 12 tests

| Algorithm | Best test accuracy | Weakest case |
|---|---:|---|
| Perceptron | 100% (Adelie vs Chinstrap, CulmenLength + CulmenDepth) | 70% (Adelie vs Chinstrap, FlipperLength + BodyMass) |
| ADALINE | 100% (Chinstrap vs Gentoo, CulmenDepth + BodyMass) | 60% (Adelie vs Chinstrap, FlipperLength + BodyMass) |

### Install and run

```bash
cd task-1
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1  |  macOS / Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt

python app.py                      # launch the GUI
python -m pytest -q                # run the tests
python -m src.experiment_runner    # regenerate results and plots
```

See [`task-1/README.md`](task-1/README.md) for details.

---

## Task 2 — Back-Propagation on a Multi-Layer Neural Network

A configurable multi-layer feedforward network written in NumPy that classifies penguins into **Adelie**, **Chinstrap** or **Gentoo** from five inputs: `CulmenLength`, `CulmenDepth`, `FlipperLength`, `OriginLocation`, `BodyMass`.

**Highlights**

* Any number of hidden layers (1–6 in the GUI), each with its own width
* Sigmoid or Tanh activation, optional bias, seeded and reproducible
* Online (stochastic) gradient descent with back-propagation, MSE loss
* Fixed split from the assignment: first 30 rows per class train (90), next 20 test (60)
* Preprocessing (imputation, standardization, category coding) fitted on training data only
* Manual confusion matrix and accuracy (no scikit-learn metrics)
* Back-propagation verified against numerical gradients in the test suite
* Experiment runner that trains 32 configurations and generates the report

**Best results on the fixed test split** (seed 42; full grid in [`task-2/report/REPORT.md`](task-2/report/REPORT.md))

| Activation | Architecture | Learning rate | Epochs | Train accuracy | Test accuracy |
|---|---|---:|---:|---:|---:|
| Sigmoid | 5-8-3 | 0.01 | 1000 | 100.00% | 98.33% |
| Tanh | 5-8-3 | 0.005 | 1000 | 98.89% | 96.67% |

> The grid picks its winner on the test set, so these numbers are optimistic. The report also includes a validation-selected check (`report/independent_evaluation.csv`) and documents limitations such as the ordinal coding of `OriginLocation`.

### Install and run

```bash
cd task-2

# create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS / Linux:         source .venv/bin/activate

python -m pip install -r requirements.txt

python app.py                      # launch the GUI
python -m pytest -q                # run the tests
python -m src.experiment_runner    # re-run all experiments and regenerate report/
```

> Run all commands from inside `task-2/`. The code imports from `src`, so running them from the repository root will fail.

---

## Notes

* Datasets are included so every task runs out of the box.
* Reports and screenshots live in each task's `report/` folder.
* Generated files (`__pycache__`, `.pytest_cache`, virtual environments, GUI "latest" plots) are excluded via the root `.gitignore`.