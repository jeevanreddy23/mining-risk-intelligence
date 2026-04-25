# White Reality Check

## Purpose
This test checks whether the best-performing model appears to beat the benchmark after adjusting for model selection / data snooping.

## Settings
- benchmark: `logistic_regression`
- target column: `target_label`
- CV folds: 5
- bootstrap iterations: 500

## Result
- observed statistic: 0.030363
- bootstrap p-value: 0.038000

## Interpretation
Low p-value suggests the best candidate outperforms the benchmark beyond data-snooping effects.

## Caution
- This is still a prototype validation layer.
- The final training table includes synthetic proxy labels and synthetic operational features.
- A strong result here does not replace site validation or mine-grade holdout testing.
