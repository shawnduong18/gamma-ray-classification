# Gamma-Ray Classification: Logistic Regression vs. Support Vector Machines

## Overview

Classification study comparing logistic regression and linear support vector
machines on the MAGIC Gamma Telescope dataset. The goal is to separate gamma-ray
events from hadronic background using geometric and intensity features of
shower images.

**Key finding:** both model families land at roughly **79% cross-validated
accuracy**, about **14 points above the 64.8% majority-class baseline**. The gap
between the two models is smaller than a quarter of a percentage point on 19,020
observations — they are not meaningfully different. That is the result: the
ceiling is set by the *linear* decision boundary itself, not by which linear
model you fit or how you tune it.

## Dataset

- **Source:** [UC Irvine ML Repository — MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159)
- **Size:** 19,020 observations, 10 numeric predictors
- **Classes:** gamma 64.8% (positive) / hadron 35.2% (negative) — *imbalanced*
- **Baseline:** predicting "gamma" for every event scores **0.648**. Every number
  below should be read against that, not against 50%.
- **Features:** fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans,
  fAlpha, fDist — all describing the shape and intensity of shower images from
  Monte Carlo simulation.

## Methodology

### Hypotheses

1. No single predictor separates the classes cleanly, so a multivariate model is required.
2. The maximal-margin classifier (SVM) will generalize at least as well as logistic
   regression, but the gap will be small.

### Design

| Choice | Detail | Why |
|---|---|---|
| Cross-validation | 5-fold stratified, fixed seed | Preserves the 65/35 class ratio in every fold |
| Scaling | `StandardScaler` inside a `Pipeline` | Scaler is fit on training folds only — no leakage into validation |
| Regularization | L2 sweep, λ = 0.1 → 10, log-spaced (10 values) | Two orders of magnitude, enough to see where train/CV separates |
| Threshold-free metric | AUC-ROC on out-of-fold scores | Comparable before an operating point is chosen |
| Confusion matrices | `cross_val_predict` | Each observation scored by the fold that held it out |
| Final coefficients | Refit on all 19,020 rows, standardized | Magnitudes directly comparable across features |

### Results

| Method | Best λ | CV accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 5.995 | **0.7905** | 0.8986 | 0.5911 | 0.8021 | 0.8476 |
| Linear SVM | 0.100 | 0.7889 | 0.8996 | 0.5845 | 0.7997 | 0.8467 |

Logistic regression finishes marginally ahead, but a 0.0016 difference on 19,020
observations is noise, not a ranking.

### Interpretation

- **Regularization barely matters.** Accuracy moves ±0.3% across two orders of
  magnitude of λ. With ~15,000 training points per fold and only 10 features, the
  linear boundary is already stable — there is nothing for the penalty to fix.
- **No meaningful overfitting.** The train/CV gap stays small throughout, which is
  what the sample-to-feature ratio predicts.
- **The models are biased toward the majority class.** Both catch ~90% of gammas
  but only ~59% of hadrons, so roughly 41% of background events pass through as
  false positives. An unweighted fit on a 65/35 split learns to favour the common
  class. Class weights or a shifted threshold would trade sensitivity for
  specificity; which direction is correct depends on whether contamination or
  missed detections are the more expensive error.

### Feature importance

Standardized logistic-regression coefficients from the final fit:

| Rank | Feature | Coefficient |
|---|---|---|
| 1 | fLength | −1.254 |
| 2 | fAlpha | −1.179 |
| 3 | fConc1 | −0.600 |
| 4 | fM3Long | +0.365 |
| 5 | fSize | −0.304 |

Negative coefficients push toward hadron, positive toward gamma. fLength and
fAlpha dominate; fM3Trans, fConc, fAsym, and fDist contribute almost nothing.

These are the features the model *leans on*, which is not the same as the
features that *cause* the outcome — correlated predictors and encoding choices
both affect the ranking.

## Known limitations

- **The SVM fits originally did not converge.** `LinearSVC` was called with
  `dual=True`, which is intended for the wide case where features outnumber
  samples. With n = 19,020 and p = 10 the solver hit `max_iter` on nearly every
  fit — visible as an SVM accuracy identical to four decimal places across all
  ten λ values. The notebook now uses `dual=False`; the SVM figures should be
  regenerated from a clean run.
- **No held-out test set.** All reported accuracy is cross-validated, and the
  final model is refit on the full dataset to report coefficients. This is a
  reasonable design for a comparison study, but the numbers are CV estimates,
  not test-set performance.
- **Simulated data.** The MAGIC dataset is Monte Carlo, so real-instrument
  performance would differ.

## Files

```
├── gamma_ray_analysis.ipynb    ← full analysis: CV sweep, metrics, ROC, final model
└── MAGIC_Gamma_Telescope.csv   ← raw data (or fetched via ucimlrepo at runtime)
```

## Installation & usage

```bash
git clone https://github.com/shawnduong18/gamma-ray-classification.git
cd gamma-ray-classification
pip install -r requirements.txt
jupyter notebook gamma_ray_analysis.ipynb   # Kernel → Restart & Run All
```

## Future work

- Non-linear models (RBF SVM, gradient boosting) to test whether the ~79% ceiling
  is really the linear boundary rather than the feature set.
- Class weights or threshold tuning to lift specificity.
- Polynomial and interaction terms to give the linear models more capacity.
- Error analysis on the misclassified events to characterise the failure modes.

## References

1. Goenawan, J., Duong, S., Han, U. (2025). *Classifying Air Showers with Logistic
   Regression and Support Vector Machines.* COGS 109 Final Project, UC San Diego.
2. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An Introduction to
   Statistical Learning.* Springer.
3. UC Irvine Machine Learning Repository. MAGIC Gamma Telescope Dataset.
   https://archive.ics.uci.edu/dataset/159

<!--
AFTER YOU RE-RUN THE NOTEBOOK, UPDATE TWO THINGS IN THIS FILE:
 1. The "Linear SVM" row of the results table — dual=False will shift those
    numbers slightly, and the best-lambda will probably no longer be 0.100.
 2. Add the AUC-ROC values printed by the new ROC cell (a row or a sentence
    under Results). Your resume claims AUC-ROC, so this needs to exist here.
This comment does not render on GitHub. Delete it once both are done.
-->
