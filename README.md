# Gamma-Ray Classification: Logistic Regression vs. Support Vector Machines

## Overview

Classification study comparing logistic regression and support vector machine 
performance on the MAGIC Gamma Telescope dataset. The goal is to distinguish 
gamma-ray events from hadronic background using geometric and intensity features 
of shower images.

**Key Finding:** Both models achieve ~79% accuracy with linear decision boundaries. 
Linear SVM marginally outperforms logistic regression (79.05% vs 79.89% CV accuracy), 
but the gap is small suggesting the performance ceiling is set by the linear model 
class itself, not the fitting procedure.

## Dataset

- **Source:** [UC Irvine Machine Learning Repository - MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159)
- **Size:** 19,020 observations, 10 numeric predictors
- **Classes:** Gamma rays (65%, positive) vs. Hadrons (35%, negative) _imbalanced_
- **Features:** fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist
  - All describe shape and intensity of shower images from Monte Carlo simulation

## Methodology

### Hypothesis

1. No single predictor cleanly separates classes → multivariate models required
2. Maximal-margin classifier (SVM) will generalize ≥ logistic regression, but gap is small

### Model Selection & Evaluation

- **Cross-validation:** 5-fold stratified (preserves class ratio in each fold)
- **Regularization:** L2 penalty sweep (λ = 0.1 to 10, log-spaced)
- **Metrics:** Accuracy, sensitivity, specificity, precision, F1, AUC-ROC, confusion matrices
- **Libraries:** Scikit-learn, Pandas, Matplotlib, Seaborn

### Key Results

| Method | Best λ | CV Accuracy | Sensitivity | Specificity | F1 |
|--------|--------|-------------|-------------|-------------|-----|
| Logistic Regression | 5.995 | 0.7905 | 0.8986 | 0.5911 | 0.8476 |
| Linear SVM | 0.100 | 0.7889 | 0.8996 | 0.5845 | 0.8467 |

**Interpretation:**
- Regularization strength has minimal impact (±0.3% accuracy across two orders of magnitude)
- Train/CV gap is small → little to no overfitting with 15,000 training points per fold
- Both models catch most gammas (high sensitivity) but miss some hadrons (lower specificity)
- SVM trades slight sensitivity loss for better specificity/precision

### Feature Importance

After refitting on full dataset, standardized logistic regression coefficients show:

1. **fLength**  strongest predictor (negative → hadron)
2. **fAlpha**  second strongest (negative → hadron)
3. **fConc1**  moderate importance
4. Others: fSize, fM3Long contribute; fM3Trans, fDist, fAsym contribute minimally

[See Figure 5 in analysis for full coefficient plot]

## Files

- `gamma_ray_analysis.ipynb`  Full analysis, visualizations, model fitting
- `MAGIC_Gamma_Telescope.csv`  Raw data from UCI repo

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/shawnduong18/gamma-ray-classification.git
cd gamma-ray-classification

# Install dependencies
pip install -r requirements.txt

# Run the analysis
jupyter notebook notebooks/gamma_ray_analysis.ipynb
```

## Key Takeaways

1. **Multivariate classification is necessary**: class overlap on every single feature 
   rules out simple threshold-based approaches
   
2. **Linear models saturate at ~79% accuracy**: both LR and SVM hit the same ceiling, 
   suggesting non-linear decision boundaries would be needed to improve further
   
3. **Regularization robustness**: accuracy is stable across λ from 0.1 to 10, indicating 
   the feature set and sample size (15K per fold) provide a stable linear boundary
   
4. **Sensitivity-specificity tradeoff**: both models prefer to predict gamma (positive class), 
   catching 90% of true gammas but letting ~41% of hadrons through as false positives

## Future Work

- Test non-linear models (RBF SVM, neural networks) to explore non-linear decision boundaries
- Address class imbalance (SMOTE, class weights) to improve specificity
- Feature engineering: create polynomial/interaction terms to boost linear model capacity
- Investigate misclassified examples to understand failure modes

## References

1. Goenawan, J., Duong, S., Han, U. (2025). "Classifying Air Showers with Logistic 
   Regression and Support Vector Machines." COGS 109 Final Project.
   
2. James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An Introduction to 
   Statistical Learning with Applications in R.* Springer.
   
3. UC Irvine Machine Learning Repository. MAGIC Gamma Telescope Dataset.
   https://archive.ics.uci.edu/dataset/159
