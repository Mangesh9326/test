import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/23_Sharktank.csv")

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Value Heatmap")
plt.show()

df.fillna(df.mean(numeric_only=True), inplace=True)

df.drop(['brand_name', 'idea'], axis=1, inplace=True)

# Step 2: Box Plot for Outliers
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
    plt.show()
    
# Step 3: Feature Scaling
target_col = 'deal' # Binary classification: 1 = got deal, 0 = no deal
X = df.drop(target_col, axis=1)
y = df[target_col]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply SMOTE to Balance Classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 5: Classification Algorithms
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "GaussianNB": GaussianNB(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
    }
# Evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    print(f"\n{name}")
    print("-" * 50)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

# Plot ROC Curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Step 6: PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.title("PCA Projection (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Step 7: Linear Regression
if df[target_col].nunique() > 2:
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    print("Linear Regression R^2 Score:", reg.score(X_test, y_test))

# Step 8: Bagging & Boosting
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50)
bag.fit(X_train, y_train)
print("\nBagging Accuracy:", accuracy_score(y_test, bag.predict(X_test)))
boost = AdaBoostClassifier(n_estimators=50)
boost.fit(X_train, y_train)
print("Boosting Accuracy:", accuracy_score(y_test, boost.predict(X_test)))

# Step 9: Cross-Validation
print("\nCross-Validation Scores (5-Fold):")
for name, model in models.items():
    scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
    print(f"{name}: Mean = {scores.mean():.4f}, Std = {scores.std():.4f}")
