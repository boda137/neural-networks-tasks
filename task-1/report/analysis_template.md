#  Results and Analysis

## 1. Experimental setup
- Dataset: supplied `penguins.csv`, containing 150 rows and 3 species.
- Each experiment selects two species and two features.
- Split: 30 training samples and 20 testing samples from each selected species (60 training, 40 testing samples total).
- Preprocessing: missing feature values are imputed using the training-set mean; selected features are standardized using statistics calculated from training data only.
- Algorithms: Single-Layer Perceptron and ADALINE, both implemented from scratch with NumPy.
- Evaluation: manually implemented confusion matrix and accuracy; no scikit-learn metrics are used.
- Reproducibility: fixed random seeds are used by the experiment runner.

**Encoding note:** `OriginLocation` is encoded as Torgersen=0, Biscoe=1, Dream=2. This ordinal encoding creates an artificial numerical order, so results using this feature should be interpreted cautiously. It is included to meet the requirement to preprocess the feature, not because the locations naturally have numeric distances.

## 2. Perceptron results
Confusion-matrix layout in the table is `[[TN, FP], [FN, TP]]`, where class 1 is encoded as -1 and class 2 as +1.

| # | Classes (-1 vs +1) | Features | Test accuracy | Confusion matrix | Training observation |
|---|---|---|---:|---|---|
| 1 | Adelie vs Gentoo | CulmenLength, FlipperLength | 97.5% | `[[19, 1], [0, 20]]` | 0 errors in the final training epoch |
| 2 | Adelie vs Chinstrap | CulmenLength, CulmenDepth | 100.0% | `[[20, 0], [0, 20]]` | 0 errors in the final training epoch |
| 3 | Chinstrap vs Gentoo | CulmenDepth, BodyMass | 100.0% | `[[20, 0], [0, 20]]` | 0 errors in the final training epoch |
| 4 | Adelie vs Gentoo | CulmenDepth, BodyMass | 97.5% | `[[20, 0], [1, 19]]` | 0 errors in the final training epoch |
| 5 | Adelie vs Chinstrap | FlipperLength, BodyMass | 70.0% | `[[16, 4], [8, 12]]` | 16 training errors in the final epoch; this combination was difficult for this linear classifier/split |

## 3. ADALINE results
Confusion-matrix layout is the same as above.

| # | Classes (-1 vs +1) | Features | Test accuracy | Confusion matrix | Final training MSE |
|---|---|---|---:|---|---:|
| 1 | Adelie vs Gentoo | CulmenLength, FlipperLength | 97.5% | `[[19, 1], [0, 20]]` | 0.22677 |
| 2 | Adelie vs Chinstrap | CulmenLength, CulmenDepth | 97.5% | `[[20, 0], [1, 19]]` | 0.35929 |
| 3 | Chinstrap vs Gentoo | CulmenDepth, BodyMass | 100.0% | `[[20, 0], [0, 20]]` | 0.13202 |
| 4 | Adelie vs Gentoo | CulmenDepth, BodyMass | 97.5% | `[[20, 0], [1, 19]]` | 0.10650 |
| 5 | Adelie vs Chinstrap | FlipperLength, BodyMass | 60.0% | `[[13, 7], [9, 11]]` | 0.67175 |

## 4. Discussion

### Best observed combinations
- **Perceptron:** the two combinations `Adelie vs Chinstrap` with `CulmenLength + CulmenDepth`, and `Chinstrap vs Gentoo` with `CulmenDepth + BodyMass`, both achieved 100% test accuracy in this experiment set.
- **ADALINE:** `Chinstrap vs Gentoo` with `CulmenDepth + BodyMass` achieved 100% test accuracy.
- The `CulmenDepth + BodyMass` pair performed strongly for Adelie/Gentoo as well, with 97.5% accuracy for both models.

### Weakest observed combination
`FlipperLength + BodyMass` for `Adelie vs Chinstrap` was the weakest tested combination: 70% accuracy for Perceptron and 60% for ADALINE. This indicates that feature usefulness depends on which classes are being separated; a feature pair that works well for one class pair is not automatically best for another.

### Perceptron vs ADALINE
Perceptron directly corrects misclassified training examples. ADALINE instead learns continuous linear outputs by minimizing mean squared error, then thresholds the output to produce a class label. On these five fixed experiments, Perceptron did better on the difficult fifth combination, while both algorithms reached perfect accuracy on the third combination. ADALINE's final MSE should not be confused with classification error: it measures the difference between target values and continuous outputs, not the fraction of incorrectly classified samples.

## 5. Limitations
1. Each combination uses one fixed train/test split, so results may change with a different split. These measurements are not a guarantee of generalization.
2. Each test set contains only 40 samples; a single error changes accuracy by 2.5 percentage points.
3. `OriginLocation` uses ordinal encoding and may introduce an artificial relationship between categories.
4. Standardization is fitted on training data only, avoiding leakage, and missing values are imputed rather than dropping rows.
5. The two algorithms use linear decision boundaries. Some feature/class combinations may not be well separated by a straight line.
6. Accuracy alone does not describe all error types; the confusion matrix is included to show false positives and false negatives.

## 6. Generated outputs
Run `python -m src.experiment_runner` to regenerate `experiment_results.csv` and ten decision-boundary plots in this folder. The plots show training points as circles, test points as crosses, and the learned decision boundary. For the submission, include a screenshot of the GUI and select representative good and weak plots, captioned with the corresponding classes and features.
