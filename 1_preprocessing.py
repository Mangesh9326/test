import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("data/1_travel.csv")
print("Original Shape:", df.shape)

df = df.drop_duplicates()
from scipy.stats import zscore
z_scores = df.select_dtypes(include=[np.number]).apply(zscore)
df = df[(np.abs(z_scores) < 3).all(axis=1)]
print("Shape after removing noise:", df.shape)

print("\nNull values in each column:\n", df.isnull().sum())
df = df.fillna(df.mean(numeric_only=True))
df = df.fillna("Unknown")

print("\nCorrelation Matrix:")
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    lower = df[col].quantile(0.05)
    upper = df[col].quantile(0.95)
    df[col] = df[col].clip(lower, upper)

print("\nGenerating boxplots for numeric columns...")
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for {col}")
    plt.tight_layout()
    plt.show()
    
if 'Rating' in df.columns:
    df = df[df['Rating'] > 3]
print("\nFinal Cleaned DataFrame:")
print(df.info())
print(df.head())
