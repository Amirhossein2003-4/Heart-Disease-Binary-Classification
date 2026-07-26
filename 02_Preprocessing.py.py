# =========================================================
# Heart Disease Dataset - Preprocessing for Pipeline Input
# =========================================================

from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

# Input dataset path (output of initial CSV conversion)
input_path = Path(r"C:\Users\amir\Downloads\heart_d_D\heart_disease.csv")

# Preprocessed output dataset path (saved in the same folder)
output_path = input_path.parent / "heart_disease_preprocessed.csv"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(input_path)

print("=" * 80)
print("ORIGINAL DATASET SHAPE")
print(df.shape)

print("\n" + "=" * 80)
print("MISSING VALUES BEFORE IMPUTATION")
print(df.isnull().sum())

# ---------------------------------------------------------
# 1. Create binary target and immediately drop the old multiclass target
# 0 -> Healthy
# 1,2,3,4 -> Heart Disease
# ---------------------------------------------------------

df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
df.drop(columns=["num"], inplace=True)

# ---------------------------------------------------------
# 2. Impute missing values
# Missing values are only present in 'ca' and 'thal' (less than 2% of data).
# Using 'most_frequent' strategy to avoid leaking information.
# ---------------------------------------------------------

cols_with_missing = ["ca", "thal"]
imputer = SimpleImputer(strategy="most_frequent")

df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])

print("\n" + "=" * 80)
print("MISSING VALUES AFTER IMPUTATION")
print(df.isnull().sum())

# ---------------------------------------------------------
# 3. Define feature types and apply explicit type casting
# ---------------------------------------------------------

continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_cols = ["sex", "fbs", "exang", "cp", "restecg", "slope", "ca", "thal"]

# Cast continuous features to float
for col in continuous_cols:
    df[col] = df[col].astype(float)

# Cast categorical features to integer (keeping nominal numeric codes)
for col in categorical_cols:
    df[col] = df[col].astype(int)

# Cast target to integer
df["target"] = df["target"].astype(int)

# ---------------------------------------------------------
# 4. Reorder columns to place the target at the end
# ---------------------------------------------------------

feature_columns = [col for col in df.columns if col != "target"]
df = df[feature_columns + ["target"]]

# ---------------------------------------------------------
# 5. Save the preprocessed dataset
# ---------------------------------------------------------

df.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print("PREPROCESSED DATASET SHAPE")
print(df.shape)

print("\n" + "=" * 80)
print("PREPROCESSED DATASET COLUMNS")
print(df.columns.tolist())

print("\n" + "=" * 80)
print("FIRST 5 ROWS OF PREPROCESSED DATASET")
print(df.head())

print("\n" + "=" * 80)
print("BINARY TARGET DISTRIBUTION")
print(df["target"].value_counts())

print("\n" + "=" * 80)
print(f"Preprocessed dataset saved successfully to:\n{output_path}")
