import pandas as pd

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num"
]

input_path = r"C:\Users\amir\Downloads\heart+disease\processed.cleveland.data"
output_path = r"C:\Users\amir\Downloads\heart+disease\heart_disease.csv"

df = pd.read_csv(
    input_path,
    header=None,
    names=columns,
    na_values="?"
)

# تبدیل همه ستون‌ها به عددی
df = df.apply(pd.to_numeric)

# ذخیره فایل CSV در همان مسیر
df.to_csv(output_path, index=False)

print(df.info())
print(df.head())
print(f"\nCSV file saved to: {output_path}")
