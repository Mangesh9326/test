import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)
df = df[["v1", "v2"]].copy()
df.columns = ["label", "text"]

# Remove missing values and duplicate rows
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode target labels: ham = 0, spam = 1
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})
df.dropna(subset=["label"], inplace=True)
df["label"] = df["label"].astype(int)

# Display dataset information
print("Dataset Shape:", df.shape)
print(df.head())
print("Class Counts:\n", df["label"].value_counts())

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# 4. TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_nn = vectorizer.fit_transform(
    X_train
).toarray().astype(np.float32)

X_test_nn = vectorizer.transform(
    X_test
).toarray().astype(np.float32)

# Convert labels to NumPy arrays
y_train = y_train.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)
print("Training Shape:", X_train_nn.shape)
print("Testing Shape:", X_test_nn.shape)

# 5. Function to Build Neural Network
def build_model(dropout=0.5, input_size=None):
    model = Sequential([
        Dense(
            64,
            activation="relu",
            input_shape=(input_size,)
        ),
        Dropout(dropout),

        Dense(32, activation="relu"),
        Dropout(dropout),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Early Stopping Callback
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 6. Train Baseline Model on All Labelled Training Data
model = build_model(
    dropout=0.5,
    input_size=X_train_nn.shape[1]
)

model.fit(
    X_train_nn,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate baseline model
baseline_loss, baseline_accuracy = model.evaluate(
    X_test_nn,
    y_test,
    verbose=0
)
print("Baseline Test Loss:", round(baseline_loss, 4))
print(
    "Baseline Test Accuracy:",
    round(baseline_accuracy * 100, 2),
    "%"
)

# 7. Split Training Data into Labelled and Unlabelled
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
    X_train_nn,
    y_train,
    test_size=0.70,
    random_state=42,
    stratify=y_train
)
print("Labelled Samples:", len(X_labeled))
print("Unlabelled Samples:", len(X_unlabeled))

# 8. Weak Augmentation and Train SSL Model
X_labeled_weak = X_labeled + np.random.normal(
    loc=0,
    scale=0.02,
    size=X_labeled.shape
).astype(np.float32)

ssl_model = build_model(
    dropout=0.3,
    input_size=X_labeled.shape[1]
)

ssl_model.fit(
    X_labeled_weak,
    y_labeled,
    epochs=20,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

# 9. Generate Pseudo-Labels for Unlabelled Data
pseudo_probs = ssl_model.predict(
    X_unlabeled,
    verbose=0
).flatten()
threshold = 0.90

# Select high-confidence predictions
mask = (
    (pseudo_probs >= threshold)
    | (pseudo_probs <= 1 - threshold)
)

X_pseudo = X_unlabeled[mask]
y_pseudo = (
    pseudo_probs[mask] >= 0.5
).astype(np.float32)

print("Unlabelled Samples:", len(X_unlabeled))
print(
    "High-Confidence Pseudo-Labelled Samples:",
    len(X_pseudo)
)
print(
    "Remaining Unused Unlabelled Samples:",
    len(X_unlabeled) - len(X_pseudo)
)

# 10. Strong Augmentation and Combine Data
X_pseudo_strong = X_pseudo + np.random.normal(
    loc=0,
    scale=0.05,
    size=X_pseudo.shape
).astype(np.float32)

# Combine labelled and pseudo-labelled samples
X_final = np.vstack((
    X_labeled,
    X_pseudo_strong
))

y_final = np.concatenate((
    y_labeled,
    y_pseudo
))
print("Final Training Samples:", len(X_final))
print("Final Label Shape:", y_final.shape)

# 11. Train Final Model
final_model = build_model(
    dropout=0.3,
    input_size=X_final.shape[1]
)

final_model.fit(
    X_final,
    y_final,
    epochs=20,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

# 12. Evaluate and Compare Models
ssl_loss, ssl_accuracy = final_model.evaluate(
    X_test_nn,
    y_test,
    verbose=0
)

print("SSL Test Loss:", round(ssl_loss, 4))
print(
    "SSL Test Accuracy:",
    round(ssl_accuracy * 100, 2),
    "%"
)

print(
    "Baseline Accuracy:",
    round(baseline_accuracy * 100, 2),
    "%"
)

# 13. Compare Model Performance
if ssl_accuracy > baseline_accuracy:
    print("Semi-Supervised Learning Improved the Model.")

elif ssl_accuracy < baseline_accuracy:
    print("Baseline Model Performed Better.")

else:
    print("Both Models Performed Equally.")
