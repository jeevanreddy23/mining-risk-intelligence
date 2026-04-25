# Research-Backed ML Notes

This project uses a practical, lightweight model stack for tabular mining-risk classification.

## Why these models

### Random Forest
Reference:
- Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.

Why used here:
- robust to noisy and mixed-source engineering data
- captures non-linear interactions between geology, structure, gravity, seismic, and synthetic operational features
- suitable baseline for small and medium tabular datasets

### Gradient Boosting
Reference:
- Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.

Why used here:
- strong performance on structured tabular data
- effective for combined public-context and operational proxy features
- efficient enough for prototype edge-oriented workflows when model size is controlled

### Logistic Regression
Reference:
- standard interpretable linear baseline used for tabular classification

Why used here:
- provides a transparent baseline
- helps identify whether the problem genuinely benefits from non-linear models

## Selection logic
The training pipeline compares candidate models and selects the best one using `macro_f1`.

Why `macro_f1`:
- the prototype labels are imbalanced
- hazard classes should not be evaluated only by overall accuracy
- macro F1 provides a safer early-stage signal for minority-risk categories

## Important limitation
Current labels and some operational features are synthetic proxies. This means model performance reflects prototype workflow quality, not field-validated underground accuracy.
