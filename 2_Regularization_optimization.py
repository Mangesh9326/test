import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# 2. Import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# 3. Load Dataset
file_path = input(
    "Enter CSV file path (example: dataset.csv): "
).strip().strip('"')

try:
    df = pd.read_csv(file_path)
except Exception as error:
    raise SystemExit(f"Could not read CSV: {error}")

# 4. Initial Data Cleaning
df.columns = [str(col).strip() for col in df.columns]

# Remove completely empty rows and columns
df = df.dropna(axis=0, how="all")
df = df.dropna(axis=1, how="all")

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)
if df.empty:
    raise SystemExit("Dataset is empty.")

# 5. Display Dataset Information
print("\nDataset shape:", df.shape)
print("\nColumns:")
for col in df.columns:
    print(" -", col)
print("\nMissing values:\n", df.isnull().sum())

# 6. Select Target Column
target_col = input(
    "\nEnter the target/output column to predict: "
).strip()

if target_col not in df.columns:
    raise SystemExit(
        f"Target column '{target_col}' was not found."
    )

# 7. Clean Target Column
df = df.dropna(subset=[target_col])
df = df.drop_duplicates().copy()

if df[target_col].nunique() < 2:
    raise SystemExit(
        "Target must contain at least two classes."
    )

# 8. Clean String Values
for col in df.select_dtypes(
    include=["object", "string"]
).columns:
    df[col] = df[col].map(
        lambda value: value.strip()
        if isinstance(value, str)
        else value
    )

# 9. Display Target Distribution
print("\nTarget distribution:\n")
print(df[target_col].value_counts())
plt.figure(figsize=(8, 4))
df[target_col].astype(str).value_counts().plot(
    kind="bar"
)

plt.title(f"Class Distribution: {target_col}")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 10. Separate Features and Target
X = df.drop(columns=[target_col]).copy()
y_original = df[target_col].copy()

if X.shape[1] == 0:
    raise SystemExit("No feature columns remain.")

# 11. Process Date Columns
for col in list(X.columns):
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X = X.drop(columns=[col])

# 12. Convert Numeric-Like Object Columns
for col in X.select_dtypes(
    include=["object", "string"]
).columns:
    converted = pd.to_numeric(
        X[col],
        errors="coerce"
    )
    if converted.notna().mean() >= 0.90:
        X[col] = converted

# Remove columns containing only missing values
X = X.dropna(axis=1, how="all")

# 13. Encode Target Labels
encoder = LabelEncoder()
y = encoder.fit_transform(
    y_original.astype(str)
)

class_names = [
    str(value) for value in encoder.classes_
]
n_classes = len(class_names)

print("\nTarget class encoding:")
for index, class_name in enumerate(class_names):
    print(f"{class_name} -> {index}")
