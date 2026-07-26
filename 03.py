# =========================================================
# 03_Logistic_Regression_Benchmarking.py
# Fair Comparison: Vanilla vs. Tuned L2, L1, Elastic Net
# =========================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split, GridSearchCV, learning_curve,
    validation_curve, StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------
DATA_PATH = Path(r"C:\Users\amir\Downloads\heart_d_D\heart_disease_preprocessed.csv")
OUTPUT_DIR = DATA_PATH.parent / "model_results"
PLOTS_DIR = OUTPUT_DIR / "diagnostic_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Data Split & Preprocessor
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)
X, y = df.drop(columns=["target"]), df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
nominal_features   = ["cp", "restecg", "thal"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), continuous_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), nominal_features)
], remainder="passthrough")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Expanded C range for all regularized models
C_RANGE = [0.001, 0.01, 0.1, 1, 10, 100]

# ---------------------------------------------------------
# 3. Model Configs
# ---------------------------------------------------------
model_configs = {
    "Vanilla": {
        "estimator": LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000, random_state=SEED),
        "param_grid": None
    },
    "Ridge (L2)": {
        "estimator": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED),
        "param_grid": {"classifier__C": C_RANGE}
    },
    "Lasso (L1)": {
        "estimator": LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED),
        "param_grid": {"classifier__C": C_RANGE}
    },
    "Elastic Net": {
        "estimator": LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, random_state=SEED),
        "param_grid": {
            "classifier__C": C_RANGE,
            "classifier__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }
}

# ---------------------------------------------------------
# 4. Train / Tune  (CV on train only — test set untouched)
# ---------------------------------------------------------
# Hyperparameters were selected using 5-fold cross-validation on the training set.
# The held-out test set was used only for final performance evaluation.

fitted_pipelines = {}
best_params_log  = {}
cv_results_log   = {}   # stores full GridSearchCV.cv_results_ for later inspection

for name, cfg in model_configs.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", cfg["estimator"])])
    if cfg["param_grid"] is None:
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        best_params_log[name]  = "N/A (no hyperparameters)"
    else:
        print(f"Tuning {name}...")
        grid = GridSearchCV(
            pipe,
            cfg["param_grid"],
            scoring="roc_auc",
            cv=cv_strategy,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
        )
        grid.fit(X_train, y_train)
        fitted_pipelines[name] = grid.best_estimator_
        best_params_log[name]  = grid.best_params_
        cv_results_log[name]   = pd.DataFrame(grid.cv_results_)
        cv_results_log[name].to_csv(
            OUTPUT_DIR / f"cv_results_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv",
            index=False
        )
        print(f"  Best: {grid.best_params_}")

# ---------------------------------------------------------
# 5. Evaluate all models on test set (reporting only)
# ---------------------------------------------------------
comparison_metrics = []
proba_store        = {}

plt.figure(figsize=(10, 8))
pr_fig, pr_ax = plt.subplots(figsize=(10, 8))

for name, pipe in fitted_pipelines.items():
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    proba_store[name] = y_proba

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    comparison_metrics.append({
        "Model":                name,
        "Accuracy":             accuracy_score(y_test, y_pred),
        "Recall (Sensitivity)": recall_score(y_test, y_pred),
        "Specificity":          tn / (tn + fp),
        "Precision":            precision_score(y_test, y_pred),
        "F1-Score":             f1_score(y_test, y_pred),
        "ROC-AUC":              roc_auc_score(y_test, y_proba),
        "Avg-Precision":        average_precision_score(y_test, y_proba),
        "Brier-Score":          brier_score_loss(y_test, y_proba),
        "Best_Params":          best_params_log[name]
    })

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix: {name}")
    fig.savefig(PLOTS_DIR / f"cm_{name.lower().replace(' ', '_').replace('(','').replace(')','')}.png", dpi=300)
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(1)
    plt.plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(y_test, y_proba):.3f}")

    # PR
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_proba)
    pr_ax.plot(rec_vals, prec_vals, label=f"{name}  AP={average_precision_score(y_test, y_proba):.3f}")

# Finalize ROC
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models\n"
          "(Hyperparameters selected via 5-fold CV on training set;\n"
          "test set used only for final evaluation)")
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.savefig(PLOTS_DIR / "roc_comparison.png", dpi=300); plt.close()

# Finalize PR
pr_ax.set_xlabel("Recall"); pr_ax.set_ylabel("Precision")
pr_ax.set_title("Precision-Recall Curves — All Models")
pr_ax.legend(fontsize=8); pr_ax.grid(alpha=0.3)
pr_fig.savefig(PLOTS_DIR / "pr_comparison.png", dpi=300); plt.close(pr_fig)

# ---------------------------------------------------------
# 6. Summary Table  (no winner selection — all models reported)
# ---------------------------------------------------------
summary_df = pd.DataFrame(comparison_metrics).set_index("Model")
summary_df = summary_df.sort_values(by="ROC-AUC", ascending=False)
summary_df.to_csv(OUTPUT_DIR / "final_model_comparison.csv")

# ---------------------------------------------------------
# 7. Feature Importance from each tuned model
# ---------------------------------------------------------
for name, pipe in fitted_pipelines.items():
    feat_names_raw = pipe.named_steps["preprocessor"].get_feature_names_out()
    coeffs         = pipe.named_steps["classifier"].coef_[0]

    feat_names_clean = (
        pd.Series(feat_names_raw)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("remainder__", "", regex=False)
    )

    imp_df = pd.DataFrame({
        "Feature":       feat_names_clean,
        "Coefficient":   coeffs,
        "Abs_Coeff":     np.abs(coeffs),
        "Odds_Ratio":    np.exp(coeffs)
    }).sort_values("Abs_Coeff", ascending=False)
    imp_df["Direction"] = np.where(imp_df["Odds_Ratio"] > 1, "Increases Risk", "Protective")

    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    imp_df.to_csv(OUTPUT_DIR / f"feature_importance_{safe_name}.csv", index=False)

    top = imp_df.head(15).sort_values("Coefficient")
    colors = ["#d62728" if c > 0 else "#1f77b4" for c in top["Coefficient"]]
    plt.figure(figsize=(9, 7))
    plt.barh(top["Feature"], top["Coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coefficient (log-odds)")
    plt.title(f"Top Features — {name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"feature_importance_{safe_name}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 8. Learning Curves (all models, on X_train/y_train only)
# ---------------------------------------------------------
for name, pipe in fitted_pipelines.items():
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=cv_strategy, scoring="roc_auc",
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8)
    )
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score")
    plt.plot(train_sizes, np.mean(val_scores,   axis=1), 'o-', label="CV Score")
    plt.title(f"Learning Curve — {name}")
    plt.xlabel("Training Samples"); plt.ylabel("ROC-AUC")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(PLOTS_DIR / f"learning_curve_{safe_name}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 9. Validation Curves (Bias–Variance vs. Regularization Strength C)
# ---------------------------------------------------------
# Shows whether regularization was actually necessary: if train and
# CV scores converge at high C, overfitting was not a serious concern
# for this dataset; a gap indicates regularization helps.

validation_curve_models = {
    "Ridge (L2)": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED),
    "Lasso (L1)": LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED),
}

# Elastic Net validation curve: fix l1_ratio at its tuned best, vary C
if "Elastic Net" in best_params_log and isinstance(best_params_log["Elastic Net"], dict):
    best_l1_ratio = best_params_log["Elastic Net"]["classifier__l1_ratio"]
    validation_curve_models["Elastic Net"] = LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=10000,
        l1_ratio=best_l1_ratio, random_state=SEED
    )

for name, estimator in validation_curve_models.items():
    vc_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])

    train_scores, val_scores = validation_curve(
        vc_pipe, X_train, y_train,
        param_name="classifier__C", param_range=C_RANGE,
        cv=cv_strategy, scoring="roc_auc", n_jobs=-1
    )

    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.figure(figsize=(8, 6))
    plt.plot(C_RANGE, np.mean(train_scores, axis=1), 'o-', label="Training ROC-AUC")
    plt.plot(C_RANGE, np.mean(val_scores,   axis=1), 'o-', label="Validation ROC-AUC")
    plt.xscale("log")
    plt.xlabel("C (Inverse Regularization Strength, log scale)")
    plt.ylabel("ROC-AUC")
    plt.title(f"Validation Curve — {name}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(PLOTS_DIR / f"validation_curve_{safe_name}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 10. Coefficient Paths (how coefficients evolve as C changes)
# ---------------------------------------------------------
coeff_path_models = {
    "Ridge (L2)": LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=SEED),
    "Lasso (L1)": LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED),
}
if "Elastic Net" in validation_curve_models:
    coeff_path_models["Elastic Net"] = validation_curve_models["Elastic Net"]

for name, estimator in coeff_path_models.items():
    coeff_records = []
    feat_names_clean = None

    for C_val in C_RANGE:
        estimator.set_params(C=C_val)
        cp_pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        cp_pipe.fit(X_train, y_train)

        if feat_names_clean is None:
            raw_names = cp_pipe.named_steps["preprocessor"].get_feature_names_out()
            feat_names_clean = (
                pd.Series(raw_names)
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
                .str.replace("remainder__", "", regex=False)
            )

        coeffs = cp_pipe.named_steps["classifier"].coef_[0]
        for feat, coef in zip(feat_names_clean, coeffs):
            coeff_records.append({"C": C_val, "Feature": feat, "Coefficient": coef})

    path_df = pd.DataFrame(coeff_records)
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path_df.to_csv(OUTPUT_DIR / f"coefficient_path_{safe_name}.csv", index=False)

    # Plot only the top features (by max |coefficient| across the path) to keep it readable
    top_features = (
        path_df.groupby("Feature")["Coefficient"]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    plt.figure(figsize=(9, 7))
    for feat in top_features:
        subset = path_df[path_df["Feature"] == feat]
        plt.plot(subset["C"], subset["Coefficient"], marker='o', label=feat)
    plt.xscale("log")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("C (Inverse Regularization Strength, log scale)")
    plt.ylabel("Coefficient (log-odds)")
    plt.title(f"Coefficient Path — {name} (Top 10 Features)")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"coefficient_path_{safe_name}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 11. Calibration Curves
# ---------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
for name, y_proba in proba_store.items():
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    brier = summary_df.loc[name, "Brier-Score"]
    plt.plot(mean_pred, frac_pos, marker='o', label=f"{name}  Brier={brier:.3f}")
plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction of Positives")
plt.title("Calibration Curves — All Models")
plt.legend(fontsize=8); plt.grid(alpha=0.3)
plt.savefig(PLOTS_DIR / "calibration_curves.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 12. Console Summary
# ---------------------------------------------------------
print("\n" + "=" * 65)
print("BENCHMARKING RESULTS")
print("Hyperparameters selected via 5-fold CV on training set.")
print("Held-out test set used only for final performance evaluation.")
print("=" * 65)
display_cols = [c for c in summary_df.columns if c != "Best_Params"]
print(summary_df[display_cols].round(4).to_string())
print("\nBest hyperparameters per model:")
for name, params in best_params_log.items():
    print(f"  {name}: {params}")
print(f"\nAll results saved in: {OUTPUT_DIR}")
# =========================================================
# 13. Detailed Analysis of the Best Model (Ridge L2)
# =========================================================

best_model = fitted_pipelines["Ridge (L2)"]

# -------------------------------
# Feature Names
# -------------------------------
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

feature_names = (
    pd.Series(feature_names)
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("remainder__", "", regex=False)
)

# -------------------------------
# Coefficients
# -------------------------------
coefficients = best_model.named_steps["classifier"].coef_[0]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients),
    "Odds_Ratio": np.exp(coefficients)
})

importance_df["Direction"] = np.where(
    importance_df["Coefficient"] > 0,
    "Increase Risk",
    "Decrease Risk"
)

importance_df = importance_df.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "ridge_feature_importance.csv",
    index=False
)

print("\n")
print("="*70)
print("TOP FEATURES (RIDGE)")
print("="*70)
print(importance_df.head(20).to_string(index=False))

# =========================================================
# Coefficient Plot
# =========================================================

top = importance_df.head(15).sort_values("Coefficient")

colors = [
    "red" if x > 0 else "steelblue"
    for x in top["Coefficient"]
]

plt.figure(figsize=(9,7))

plt.barh(
    top["Feature"],
    top["Coefficient"],
    color=colors
)

plt.axvline(0,color="black")

plt.xlabel("Coefficient (Log Odds)")
plt.ylabel("Feature")
plt.title("Top 15 Most Important Features (Ridge Logistic Regression)")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR/"ridge_top_coefficients.png",
    dpi=300
)

plt.show()


# =========================================================
# Odds Ratio Plot
# =========================================================

top_odds = importance_df.head(15).sort_values("Odds_Ratio")

colors = [
    "red" if x>1 else "steelblue"
    for x in top_odds["Odds_Ratio"]
]

plt.figure(figsize=(9,7))

plt.barh(
    top_odds["Feature"],
    top_odds["Odds_Ratio"],
    color=colors
)

plt.axvline(1,color="black")

plt.xlabel("Odds Ratio")

plt.title("Top 15 Odds Ratios (Ridge Logistic Regression)")

plt.tight_layout()

plt.savefig(
    PLOTS_DIR/"ridge_odds_ratio.png",
    dpi=300
)

plt.show()


# =========================================================
# Predicted Probability Distribution
# =========================================================

y_proba = best_model.predict_proba(X_test)[:,1]

plt.figure(figsize=(8,6))

plt.hist(
    y_proba[y_test==0],
    bins=20,
    alpha=0.6,
    label="Negative"
)

plt.hist(
    y_proba[y_test==1],
    bins=20,
    alpha=0.6,
    label="Positive"
)

plt.xlabel("Predicted Probability")

plt.ylabel("Count")

plt.title("Predicted Probability Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR/"ridge_probability_distribution.png",
    dpi=300
)

plt.show()


# =========================================================
# ROC Curve (Best Model)
# =========================================================

fpr,tpr,_ = roc_curve(y_test,y_proba)

auc = roc_auc_score(y_test,y_proba)

plt.figure(figsize=(7,7))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ROC AUC = {auc:.3f}"
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Best Model (Ridge)")

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR/"ridge_roc.png",
    dpi=300
)

plt.show()


# =========================================================
# Precision Recall Curve
# =========================================================

precision,recall,_ = precision_recall_curve(
    y_test,
    y_proba
)

ap = average_precision_score(
    y_test,
    y_proba
)

plt.figure(figsize=(7,7))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"AP = {ap:.3f}"
)

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curve (Ridge)")

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOTS_DIR/"ridge_pr_curve.png",
    dpi=300
)

plt.show()

print("\nDetailed analysis for Ridge model completed.")