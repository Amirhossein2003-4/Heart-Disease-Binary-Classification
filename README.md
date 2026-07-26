# Heart Disease Binary Classification (UCI Cleveland Dataset)

A binary classification pipeline built with Scikit-Learn to predict the presence of coronary artery disease (CAD) from clinical and diagnostic test features, using the classic UCI Heart Disease (Cleveland) dataset.

This project focuses on a fair, methodologically rigorous benchmark of regularized logistic regression variants (Vanilla, Ridge/L2, Lasso/L1, Elastic Net), with full hyperparameter tuning via cross-validation, calibration analysis, and clinical interpretation of the resulting coefficients.

## Biological Background

Coronary artery disease is diagnosed using a mix of demographic factors (age, sex), symptoms (chest pain type, exercise-induced angina), resting measurements (blood pressure, cholesterol, resting ECG), and stress-test / imaging results (max heart rate, ST depression, ST slope, number of vessels seen on fluoroscopy, thallium scan result).

The target variable `num` originally encodes disease severity on a 0–4 scale; for this project it was collapsed to a binary label:

- `0` → No heart disease
- `1` (from original 1–4) → Heart disease present

This mirrors the standard approach used in most published work on this dataset, since class 1–4 mainly differ in severity, not in the underlying diagnostic decision (healthy vs. diseased).

## Dataset

**Source:** UCI Heart Disease Dataset (Cleveland subset)
**Samples:** 303
**Features:** 13 clinical/diagnostic variables + 1 target (`num` → `target`)

| Column | Description | Type |
|---|---|---|
| age | Age (years) | Continuous |
| sex | 1 = male, 0 = female | Binary |
| cp | Chest pain type (1–4) | Nominal |
| trestbps | Resting blood pressure (mmHg) | Continuous |
| chol | Serum cholesterol (mg/dL) | Continuous |
| fbs | Fasting blood sugar > 120 mg/dL | Binary |
| restecg | Resting ECG result (0–2) | Nominal |
| thalach | Max heart rate achieved | Continuous |
| exang | Exercise-induced angina | Binary |
| oldpeak | ST depression induced by exercise | Continuous |
| slope | Slope of peak exercise ST segment (1–3) | Nominal/Ordinal |
| ca | Number of major vessels colored by fluoroscopy (0–3) | Ordinal/Count |
| thal | Thallium stress test result (3, 6, 7) | Nominal |
| target | 0 = healthy, 1 = disease present | Binary (derived) |

Missing values were present only in `ca` and `thal` (<2% of rows) and were imputed using the most frequent value of each column to avoid introducing distributional leakage.

## Pipeline

Four logistic regression configurations were trained and compared on an identical 80/20 stratified train/test split, all using the same preprocessing (`StandardScaler` for continuous features, `OneHotEncoder(drop="first")` for nominal categorical features, remainder passthrough for binary/ordinal features).

**Model 1 — Vanilla Logistic Regression**
No penalty (`penalty=None`), `lbfgs` solver — baseline with no regularization.

**Model 2 — Ridge (L2)**
`penalty="l2"`, `lbfgs` solver, `C` tuned via `GridSearchCV` over `[0.001, 0.01, 0.1, 1, 10, 100]`.

**Model 3 — Lasso (L1)**
`penalty="l1"`, `liblinear` solver, same `C` grid — encourages sparse coefficients (feature selection).

**Model 4 — Elastic Net**
`penalty="elasticnet"`, `saga` solver, joint grid search over `C` and `l1_ratio ∈ [0.1, 0.3, 0.5, 0.7, 0.9]`.

All hyperparameters were selected using 5-fold stratified cross-validation on the **training set only**, optimizing ROC-AUC. The held-out test set was used exclusively for final evaluation — never for model selection.

## Results

| Model | Accuracy | Recall | Specificity | Precision | F1 | ROC-AUC | Avg-Precision | Brier |
|---|---|---|---|---|---|---|---|---|
| **Ridge (L2)** | 0.8852 | 0.9286 | 0.8485 | 0.8387 | 0.8814 | **0.9578** | 0.9428 | 0.0812 |
| Elastic Net | 0.8852 | 0.9286 | 0.8485 | 0.8387 | 0.8814 | 0.9578 | 0.9428 | 0.0813 |
| Lasso (L1) | 0.8689 | 0.9286 | 0.8182 | 0.8125 | 0.8667 | 0.9556 | 0.9368 | 0.0842 |
| Vanilla | 0.8689 | 0.8571 | 0.8788 | 0.8571 | 0.8571 | 0.9091 | 0.9053 | 0.1127 |

**Best hyperparameters:**
- Ridge (L2): `C = 1`
- Lasso (L1): `C = 1`
- Elastic Net: `C = 1`, `l1_ratio = 0.1`

Ridge and Elastic Net are essentially tied at the top, both outperforming Lasso and clearly outperforming the unregularized Vanilla model — most notably in ROC-AUC (0.958 vs. 0.909) and calibration (Brier score 0.081 vs. 0.113). The near-zero `l1_ratio` chosen for Elastic Net indicates the tuning process converged toward an L2-dominated solution, consistent with Ridge being the single best-performing penalty on its own.

## Feature Importance (Ridge — Best Model)

| Feature | Coefficient | Odds Ratio | Direction |
|---|---|---|---|
| thal_7 (reversible defect) | 1.314 | 3.72 | ↑ Risk |
| cp_4 (asymptomatic) | 1.284 | 3.61 | ↑ Risk |
| sex (male) | 1.228 | 3.42 | ↑ Risk |
| ca (vessels involved) | 1.190 | 3.29 | ↑ Risk |
| exang (exercise angina) | 0.620 | 1.86 | ↑ Risk |
| slope | 0.576 | 1.78 | ↑ Risk |
| thalach (max HR) | -0.408 | 0.66 | ↓ Risk |
| restecg_2 (LVH) | 0.372 | 1.45 | ↑ Risk |
| cp_2 (atypical angina) | 0.320 | 1.38 | ↑ Risk |
| trestbps | 0.274 | 1.32 | ↑ Risk |
| chol | 0.217 | 1.24 | ↑ Risk |
| fbs | -0.193 | 0.82 | ↓ Risk |
| oldpeak | 0.185 | 1.20 | ↑ Risk |
| thal_6 (fixed defect) | 0.155 | 1.17 | ↑ Risk |
| cp_3 (non-anginal pain) | -0.135 | 0.87 | ↓ Risk |
| age | -0.082 | 0.92 | ↓ Risk |

Clinically, this ranking is coherent: a reversible thallium defect (`thal_7`), asymptomatic chest pain (`cp_4` — often a red flag precisely because there is no warning symptom), male sex, and number of vessels involved (`ca`) are the strongest drivers of predicted risk, matching well-established cardiology risk factors. Max heart rate (`thalach`) is protective, as expected — a healthier heart sustains a higher rate under exertion.

The counter-intuitive negative coefficient for `age` is a known artifact of this dataset: patients referred for angiography at younger ages in this cohort tended to have more severe/obvious symptoms driving the referral, partially confounding the raw age-risk relationship once other features are controlled for.

## Diagnostics Performed

Beyond the headline metrics, the pipeline generates a full diagnostic suite for every model:

- **ROC & Precision-Recall curves** (all models overlaid)
- **Confusion matrices** per model
- **Calibration curves** (quantile-binned, with Brier scores) — Ridge/Elastic Net are the best-calibrated models
- **Learning curves** (train vs. CV ROC-AUC vs. training set size) — checks for high bias/variance
- **Validation curves** (ROC-AUC vs. `C`, log scale) — checks whether regularization strength materially matters for this dataset size
- **Coefficient paths** (how each feature's coefficient evolves as `C` varies) — visualizes the Lasso/Elastic Net shrinkage-to-zero behavior directly
- **Predicted probability distribution** split by true class, for the best model

All plots and CSVs (cross-validation results, feature importances, coefficient paths) are saved under `model_results/`.
<img width="2400" height="1800" alt="learning_curve_elastic_net" src="https://github.com/user-attachments/assets/9ae9a922-064c-44e2-be70-3d5ea24a029f" />
<img width="3000" height="2400" alt="pr_comparison" src="https://github.com/user-attachments/assets/7d8111f8-9920-4076-a198-280c28b54095" />

## Limitations

- **Small sample size (303 rows).** With this few samples, cross-validation folds are small and metric estimates carry non-trivial variance; results should be treated as indicative rather than a tight confidence interval.
- **Single-center cohort (Cleveland).** The Cleveland subset is the most complete of the original 4-hospital UCI collection but does not necessarily generalize to other populations or imaging/lab protocols.
- **Linear decision boundary.** Logistic regression (even regularized) assumes additive log-odds effects; it cannot capture interaction effects (e.g., age × cholesterol) without explicit feature engineering.
- **Median/mode imputation for missingness.** `ca` and `thal` missingness was imputed with the most frequent category; this is simple and leakage-safe but does not model uncertainty from the missing values.
- **`num` collapsed to binary.** Severity information (grades 1–4) was discarded to simplify the modeling task; an ordinal or multiclass approach would preserve more clinical information.

## Interpretation

Regularization clearly helps here: both Ridge and Elastic Net meaningfully outperform the unpenalized Vanilla model on ROC-AUC, calibration (Brier score), and F1 — indicating that, despite the small feature set, some degree of overfitting/variance reduction is beneficial at n=303. Elastic Net's own tuning converging to a near-pure-L2 solution (`l1_ratio=0.1`) is itself evidence that none of the 13 clinical features are redundant enough to warrant aggressive sparsification — all of them carry at least some signal, which is consistent with the known clinical literature on these specific risk factors. The resulting Ridge model's feature ranking recovers well-established cardiology risk factors (thallium defect, chest pain type, vessel involvement, male sex) without any manual feature engineering, giving confidence that the model is learning genuine signal rather than dataset artifacts.

## Next Steps

- Validate on external cohorts (Hungarian, Switzerland, VA Long Beach subsets from the same UCI collection) to test generalization beyond Cleveland.
- Add nonlinear models (Random Forest, XGBoost, Gradient Boosting) as a benchmark ceiling against the linear approach.
- Explore interaction terms (e.g., `age × sex`, `chol × fbs`) within the logistic regression framework before jumping to fully nonlinear models.
- Model `num` as an ordinal target (disease severity) instead of collapsing to binary.
- Apply nested cross-validation to get an unbiased estimate of generalization performance given the small sample size.
- SHAP-based interpretation to complement coefficient-based interpretation, especially useful once nonlinear models are added.

## Author

Amirhossein
Email: amirhossein070905@gmail.com
Telegram: https://t.me/itsamirhosseingadimi
