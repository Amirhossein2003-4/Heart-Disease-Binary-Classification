"""
ویژگی‌هایی که استفاده می‌کنیم

دقیقاً همین ۱۴ ستون:

ستون    /	توضیح

age	        سن
sex	        جنسیت (1=مرد، 0=زن)



cp	        نوع درد قفسه سینه
1 = Typical Angina
2 = Atypical Angina
3 = Non-anginal Pain
4 = Asymptomatic
فقط چهار گروه مختلف هستند.




trestbps	فشار خون در حالت استراحت

chol	    کلسترول سرم

fbs	قند خون ناشتا    >120

restecg   	نتیجه ECG در حالت استراحت


thalach	    حداکثر ضربان قلب



exang	    آنژین ناشی از ورزش
1 = Exercise-induced angina
0 = No exercise-induced angina

oldpeak	    میزان افت ST


slope	    شیب قطعه ST
1 = Upsloping
2 = Flat
3 = Downsloping


ca	        تعداد عروق اصلی مشاهده‌شده در فلوروسکوپی



thal     	نتیجه تست تالیم
3 = Normal
6 = Fixed Defect
7 = Reversible Defect

num        	برچسب (Target)
"""

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""Feature	       نوع
age	                Continuous
trestbps        	Continuous
chol	            Continuous
thalach	            Continuous
oldpeak	            Continuous
sex	                Binary
fbs	                Binary
exang	            Binary
cp	                Nominal
restecg	            Nominal
slope	            Nominal (یا Ordinal، بسته به رویکرد)
thal	            Nominal
ca	                Ordinal / Count
num	Target          (بعداً Binary می‌شود)"""

#==========================================================================
#==========================================================================
"""Feature	                     مفهوم زیستی / پزشکی	تفسیر کلی
age	                            سن بیمار (سال)	با افزایش سن، خطر آترواسکلروز و بیماری عروق کرونر افزایش می‌یابد.
sex	                            جنسیت (1=مرد، 0=زن)	مردان در سنین پایین‌تر بیشتر در معرض CAD هستند؛ پس از یائسگی، خطر در زنان نیز افزایش می‌یابد.
cp	                            نوع درد قفسه سینه	مهم‌ترین علامت بالینی بیماری کرونری. نوع درد می‌تواند احتمال بیماری را تغییر دهد.
trestbps	                    فشار خون سیستولیک در حالت استراحت (mmHg)	فشار خون بالا باعث آسیب اندوتلیوم، تسریع آترواسکلروز و افزایش بار قلب می‌شود.
chol	                       کلسترول تام خون (mg/dL)	کلسترول بالا، به‌ویژه LDL، یکی از عوامل اصلی تشکیل پلاک‌های آترواسکلروتیک است.
fbs	قند خون ناشتا                     >120 mg/dL	           نماینده‌ای از دیابت یا اختلال متابولیسم گلوکز؛ دیابت از مهم‌ترین عوامل خطر بیماری قلبی است.
restecg	                      نتیجه ECG در حالت استراحت	تغییرات الکتریکی قلب می‌توانند نشان‌دهنده ایسکمی، هیپرتروفی یا آسیب قلبی باشند.
thalach	                      حداکثر ضربان قلب هنگام تست ورزش	توانایی قلب برای افزایش ضربان در پاسخ به ورزش. مقادیر پایین‌تر ممکن است نشان‌دهنده محدودیت عملکرد قلب باشد.
exang	                      آنژین ایجادشده با ورزش	اگر هنگام فعالیت درد قفسه سینه ایجاد شود، احتمال بیماری عروق کرونر بیشتر است.
oldpeak	                      میزان افت قطعه ST پس از ورزش نسبت به استراحت	یکی از مهم‌ترین شاخص‌های ایسکمی میوکارد در تست ورزش؛ هرچه بیشتر باشد، احتمال بیماری بیشتر است.
slope	                       شیب قطعه ST در اوج ورزش	شکل تغییر ST اطلاعات مهمی درباره احتمال ایسکمی می‌دهد؛ Flat و Downsloping معمولاً نگران‌کننده‌تر از Upsloping هستند.
ca	                            تعداد عروق اصلی که در آنژیوگرافی با ماده حاجب دیده شده‌اند (0 تا 3)	تعداد بیشتر معمولاً نشان‌دهنده درگیری گسترده‌تر عروق کرونر است. این ویژگی معمولاً قدرت پیش‌بینی بالایی دارد.
thal	                        نتیجه اسکن پرفیوژن تالیم	3: طبیعی، 6: نقص ثابت (اغلب اسکار ناشی از انفارکتوس قبلی)، 7: نقص برگشت‌پذیر (ایسکمی قابل برگشت). از قوی‌ترین ویژگی‌های تشخیصی است.
num	                               وضعیت بیماری کرونر بر اساس آنژیوگرافی	هدف مدل. 0 یعنی عدم وجود بیماری، و 1 تا 4 درجات مختلف وجود بیماری. در اکثر مطالعات به دو کلاس «سالم» و «بیمار» تبدیل می‌شود.




برای ستون cp (نوع درد قفسه سینه)
مقدار	معنی
1	Typical Angina: درد کلاسیک ناشی از ایسکمی؛ معمولاً با فعالیت شروع و با استراحت یا نیتروگلیسیرین بهتر می‌شود.
2	Atypical Angina: برخی ویژگی‌های آنژین را دارد، اما کاملاً تیپیک نیست.
3	Non-anginal Pain: درد قفسه سینه با منشأ غیرکرونری (مثلاً عضلانی یا گوارشی).
4	Asymptomatic: بدون درد قفسه سینه؛ ممکن است بیماری به‌صورت خاموش وجود داشته باشد.




برای ستون restecg
مقدار	معنی
0	ECG طبیعی
1	ناهنجاری ST-T (می‌تواند نشانه ایسکمی یا اختلالات دیگر باشد)
2	شواهد هیپرتروفی بطن چپ (LVH) بر اساس معیارهای Estes




برای ستون slope
مقدار	معنی
1	Upsloping (شیب رو به بالا)
2	Flat (مسطح)
3	Downsloping (شیب رو به پایین)

از نظر بالینی، Flat و به‌ویژه Downsloping بیشتر با ایسکمی میوکارد ارتباط دارند.

برای ستون thal
مقدار	معنی
3	Normal Perfusion (طبیعی)
6	Fixed Defect (نقص ثابت؛ معمولاً اسکار پس از انفارکتوس)
7	Reversible Defect (نقص برگشت‌پذیر؛ نشان‌دهنده ایسکمی قابل برگشت)"""









# =====================================================
# Heart Disease Dataset - Complete EDA + Save All Figures
# =====================================================

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# Plot Style
# -----------------------------------------------------
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
file_path = r"C:\Users\amir\Downloads\heart_d_D\heart_disease.csv"
df = pd.read_csv(file_path)

# -----------------------------------------------------
# Create output folder for plots
# -----------------------------------------------------
file_dir = Path(file_path).parent
output_dir = file_dir / "eda_plots"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Plots will be saved in: {output_dir}")

# -----------------------------------------------------
# Helper function to save and show plots
# -----------------------------------------------------
plot_counter = 1

def save_and_show(title, fig=None):
    global plot_counter

    if fig is None:
        fig = plt.gcf()

    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    filename = f"{plot_counter:03d}_{safe_title}.png"
    save_path = output_dir / filename

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)

    plot_counter += 1

# -----------------------------------------------------
# Basic Information
# -----------------------------------------------------
print("=" * 80)
print("DATASET SHAPE")
print(df.shape)

print("\n" + "=" * 80)
print("FIRST 5 ROWS")
print(df.head())

print("\n" + "=" * 80)
print("DATA TYPES")
print(df.dtypes)

print("\n" + "=" * 80)
print("INFO")
df.info()

print("\n" + "=" * 80)
print("MISSING VALUES")
print(df.isnull().sum())

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print(df.describe().T)

# -----------------------------------------------------
# Duplicate rows
# -----------------------------------------------------
print("\n" + "=" * 80)
print("DUPLICATE ROWS")
print(df.duplicated().sum())

# -----------------------------------------------------
# Target Distribution
# -----------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="num")
plt.title("Target Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.tight_layout()
save_and_show("Target Distribution")

# -----------------------------------------------------
# Correlation Matrix
# -----------------------------------------------------
corr = df.corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.tight_layout()
save_and_show("Correlation Matrix")

# -----------------------------------------------------
# Histograms of all numerical columns
# -----------------------------------------------------
hist_fig = df.hist(
    figsize=(18, 16),
    bins=20,
    edgecolor="black"
)

plt.suptitle("Histograms of Numerical Features", fontsize=20)
plt.tight_layout()
save_and_show("Histograms of Numerical Features")

# -----------------------------------------------------
# Distribution + KDE
# -----------------------------------------------------
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    save_and_show(f"Distribution of {col}")

# -----------------------------------------------------
# Boxplots
# -----------------------------------------------------
for col in numeric_columns:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    save_and_show(f"Boxplot of {col}")

# -----------------------------------------------------
# Count plots for categorical variables
# -----------------------------------------------------
categorical_cols = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
    "num"
]

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Count Plot - {col}")
    plt.tight_layout()
    save_and_show(f"Count Plot {col}")

# -----------------------------------------------------
# Scatter Plot Matrix
# -----------------------------------------------------
pair_grid = sns.pairplot(
    df,
    hue="num",
    diag_kind="kde",
    corner=True
)
pair_grid.fig.suptitle("Pairplot", y=1.02)
pair_grid.fig.tight_layout()
save_and_show("Pairplot", pair_grid.fig)

# -----------------------------------------------------
# Scatter Plots Against Target
# -----------------------------------------------------
features = df.columns.drop("num")

for col in features:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col, y="num")
    plt.title(f"{col} vs Target")
    plt.tight_layout()
    save_and_show(f"{col} vs Target Scatter")

# -----------------------------------------------------
# Violin Plots
# -----------------------------------------------------
for col in features:
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df, x="num", y=col)
    plt.title(f"{col} by Target")
    plt.tight_layout()
    save_and_show(f"{col} by Target Violin")

# -----------------------------------------------------
# Boxplots grouped by Target
# -----------------------------------------------------
for col in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="num", y=col)
    plt.title(f"{col} vs Target")
    plt.tight_layout()
    save_and_show(f"{col} vs Target Boxplot")

# -----------------------------------------------------
# Pairwise Correlation with Target
# -----------------------------------------------------
corr_target = corr["num"].sort_values(ascending=False)

print("\n" + "=" * 80)
print("CORRELATION WITH TARGET")
print(corr_target)

# -----------------------------------------------------
# Skewness
# -----------------------------------------------------
print("\n" + "=" * 80)
print("SKEWNESS")
print(df.skew(numeric_only=True))

# -----------------------------------------------------
# Kurtosis
# -----------------------------------------------------
print("\n" + "=" * 80)
print("KURTOSIS")
print(df.kurtosis(numeric_only=True))

print("\nEDA Finished Successfully.")
print(f"All plots saved in: {output_dir}")
