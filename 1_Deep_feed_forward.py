import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("Depression_Student_Dataset.csv")

# 3. Data Preprocessing
# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Drop ID column if it exists
df.drop(columns=["id"], inplace=True, errors="ignore")

# Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Encode target variable
df["Depression"] = df["Depression"].map({
    "No": 0,
    "Yes": 1
})

# Encode categorical input features
X = pd.get_dummies(
    df.drop("Depression", axis=1),
    drop_first=True,
    dtype=int
)
y = df["Depression"]

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Build Feedforward Neural Network
model = Sequential([
    Dense(
        128,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),
    Dropout(0.30),
    Dense(64, activation="relu"),
    Dropout(0.20),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# 7. Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 8. Early Stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 9. Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)


# 10. Evaluate Model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

# Predict probabilities and classes
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (
    y_pred_prob >= 0.5
).astype(int).ravel()

# 11. Display Results
# Calculate training accuracy
train_accuracy = model.evaluate(
    X_train,
    y_train,
    verbose=0
)[1]
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
