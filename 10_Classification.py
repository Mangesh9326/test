import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Data Preprocessing Section ---
data = {
   'Gene': ['Gene' + str(i) for i in range(1, 11)],
   'Sample_1': np.random.rand(10),
   'Sample_2': np.random.rand(10),
   'Sample_3': np.random.rand(10),
   'Sample_4': np.random.rand(10),
   'Sample_5': np.random.rand(10),
   'Sample_6': np.random.rand(10),
}
df = pd.DataFrame(data)
df.set_index('Gene', inplace=True)
df.iloc[2, 3] = np.nan
df.iloc[5, 1] = np.nan
print("Original Dataset with Missing Values:")
print(df)
df_filled = df.apply(lambda x: x.fillna(x.mean()), axis=1)
print("\nDataset after Filling Missing Values:")
print(df_filled)
z_scores = np.abs(stats.zscore(df_filled, axis=1))
print("\nZ-scores of the data:")
print(z_scores)
df_no_outliers = df_filled.mask(z_scores > 2)
df_no_outliers = df_no_outliers.apply(lambda x: x.fillna(x.mean()), axis=1)
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_no_outliers),
                            columns=df.columns,
                            index=df.index)
print("\nNormalized Dataset:")
print(df_normalized)
df_transposed = df_normalized.T
selector = VarianceThreshold(threshold=0.01)
filtered = selector.fit_transform(df_transposed)
selected_genes = df_transposed.columns[selector.get_support()]
df_selected = pd.DataFrame(filtered.T, index=selected_genes, columns=df.columns)
print("\nAfter Feature Selection:")
print(df_selected)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_selected)
plt.figure()
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title("PCA of Gene Expression Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# --- Classification Section ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
   accuracy_score, precision_score, recall_score, f1_score,
   classification_report, ConfusionMatrixDisplay
)
RNG = 42
np.random.seed(RNG)
ds = load_breast_cancer()
X = pd.DataFrame(ds.data, columns=ds.feature_names)
y = pd.Series(ds.target, name='target')
print("\nShape:", X.shape)
print("Classes:", dict(zip(ds.target_names, [np.sum(y==i) for i in range(len(ds.target_names))])))
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, stratify=y, random_state=RNG
)
print("Train size:", X_train.shape, " Test size:", X_test.shape)
pipelines = {
   "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(random_state=RNG))]),
   "Random Forest": Pipeline([("clf", RandomForestClassifier(random_state=RNG))]),
   "SVM (RBF)": Pipeline([
       ("scaler", StandardScaler()),
       ("clf", SVC(kernel="rbf", probability=True, random_state=RNG))
   ]),
}
results = []
for name, pipe in pipelines.items():
   pipe.fit(X_train, y_train)
   y_pred = pipe.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   prec = precision_score(y_test, y_pred, zero_division=0)
   rec = recall_score(y_test, y_pred, zero_division=0)
   f1 = f1_score(y_test, y_pred, zero_division=0)
   print(f"\n=== {name} ===")
   print(classification_report(y_test, y_pred, target_names=ds.target_names, zero_division=0))
   disp = ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
   plt.title(f"{name} -- Confusion Matrix")
   plt.show()
   results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False).reset_index(drop=True)
print("\nBaseline Results:")
print(results_df)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
cv_rows = []
for name, pipe in pipelines.items():
   scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
   row = {"Model": name}
   for m in scoring.keys():
       row[f"{m}_mean"] = scores[f"test_{m}"].mean()
       row[f"{m}_std"] = scores[f"test_{m}"].std()
   cv_rows.append(row)
cv_df = pd.DataFrame(cv_rows)
print("\nCross-Validation Results:")
print(cv_df.sort_values(by="f1_mean", ascending=False).reset_index(drop=True))
param_grids = {
   "Decision Tree": {"clf__max_depth": [None, 5, 10, 20], "clf__min_samples_split": [2, 10]},
   "Random Forest": {"clf__n_estimators": [100], "clf__max_depth": [None, 10, 20], "clf__max_features": ["sqrt", "log2"]},
   "SVM (RBF)": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]},
}
best_models = {}
for name, pipe in pipelines.items():
   grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring="f1", n_jobs=-1)
   grid.fit(X_train, y_train)
   best_models[name] = grid.best_estimator_
   print(f"{name} best params:", grid.best_params_)
tuned_results = []
for name, model in best_models.items():
   y_pred = model.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   prec = precision_score(y_test, y_pred, zero_division=0)
   rec = recall_score(y_test, y_pred, zero_division=0)
   f1 = f1_score(y_test, y_pred, zero_division=0)
   print(f"\n=== Tuned {name} ===")
   print(classification_report(y_test, y_pred, target_names=ds.target_names, zero_division=0))
   disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
   plt.title(f"Tuned {name} -- Confusion Matrix")
   plt.show()
   tuned_results.append({"Model": f"Tuned {name}", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
print("\nTuned Results:")
print(pd.DataFrame(tuned_results).sort_values(by="F1", ascending=False).reset_index(drop=True))

# --- K-mers Text Extraction Section ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
def kmers(seq, k=3):
   return [seq[i:i+k] for i in range(len(seq) - k + 1)]
sequences = [
   "ATGCGTACGTTAGC", "ATGCGTACGTTAGG", "ATGCGTACGTTAAC", "ATGCGTACGTTTGC",
   "TTTTGCGCGATCAA", "TTTGGCGCGATCAC", "TTTCGCGCGATCAT", "TTTAGCGCGATCAG",
   "GGGCGTATATATGC", "GGGCGTATATATGA", "GGGCGTATATATGG", "GGGCGTATATATGT",
]
labels = [1,1,1,1, 0,0,0,0, 1,1,0,0]
k = 3
docs = [" ".join(kmers(s, k)) for s in sequences]
Xtr, Xte, ytr, yte = train_test_split(docs, labels, test_size=0.3, stratify=labels, random_state=RNG)

# FIX: Corrected the regex pattern below
pipe_seq = Pipeline([
   ("vec", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
   ("svm", LinearSVC(random_state=RNG))
])
pipe_seq.fit(Xtr, ytr)
yp = pipe_seq.predict(Xte)
acc = accuracy_score(yte, yp)
prec = precision_score(yte, yp, zero_division=0)
rec = recall_score(yte, yp, zero_division=0)
f1 = f1_score(yte, yp, zero_division=0)
print(f"\nk = {k}")
print(classification_report(yte, yp, zero_division=0))
print({"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})