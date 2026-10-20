import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 2. Set Random Seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 3. Load Dataset and Create Prediction Tasks
df = pd.read_csv(
    "student-mat.csv",
    sep=";"
)

# Create target variables
df["PassFail"] = (df["G3"] >= 10).astype(int)

df["GoodAttendance"] = (
    df["absences"] <= df["absences"].median()
).astype(int)

df["LowAcademicRisk"] = (
    df["failures"] == 0
).astype(int)

# Define target columns and display names
targets = [
    "PassFail",
    "GoodAttendance",
    "LowAcademicRisk"
]

names = [
    "Pass/Fail",
    "Attendance",
    "Academic Risk"
]

# 4. Select Input Features
num = [
    "age",
    "absences",
    "failures",
    "studytime",
    "traveltime",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health"
]

cat = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic"
]
X = df[num + cat]
Y = df[targets]

# 5. Split Dataset into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=SEED,
    stratify=Y["PassFail"]
)

# 6. Preprocess Numerical and Categorical Features
prep = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        num
    ),
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        cat
    )
])

# Fit preprocessing on training data only
X_train = prep.fit_transform(X_train)
X_test = prep.transform(X_test)

# Convert features to PyTorch tensors
X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

# Convert target labels to PyTorch tensor
Y_train = torch.tensor(
    Y_train.values,
    dtype=torch.float32
)

# 7. Define Multitask Neural Network
class MultiTaskNetwork(nn.Module):
    def __init__(self, n):
        super().__init__()

        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Separate output heads for each task
        self.pass_head = nn.Linear(32, 1)
        self.attendance_head = nn.Linear(32, 1)
        self.risk_head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.shared(x)

        return (
            self.pass_head(x),
            self.attendance_head(x),
            self.risk_head(x)
        )

# Initialize the model
model = MultiTaskNetwork(
    X_train.shape[1]
)

# 8. Define Loss Functions and Optimizer
weights = []

for i in range(3):
    neg = (Y_train[:, i] == 0).sum()
    pos = (Y_train[:, i] == 1).sum()

    # Calculate positive-class weight
    weights.append(
        neg / max(pos.item(), 1)
    )

losses = [
    nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([w])
    )
    for w in weights
]

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# 9. Train the Multitask Neural Network
loss_history = []

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)

    # Calculate combined loss across all tasks
    total_loss = sum(
        losses[i](
            outputs[i].squeeze(1),
            Y_train[:, i]
        )
        for i in range(3)
    )

    # Backpropagation and parameter update
    total_loss.backward()
    optimizer.step()

    loss_history.append(
        total_loss.item()
    )

    # Display progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss.item():.4f}"
        )

# 10. Plot Training Loss
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(loss_history) + 1),
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Multitask Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 11. Evaluate All Three Prediction Tasks
model.eval()
with torch.no_grad():
    outputs = model(X_test)

    # Convert logits to probabilities
    probs = [
        torch.sigmoid(output)
        .squeeze(1)
        .numpy()
        for output in outputs
    ]

    # Convert probabilities to binary predictions
    preds = [
        (prob >= 0.5).astype(int)
        for prob in probs
    ]

# Actual target labels
actual = [
    Y_test[target].values
    for target in targets
]
results = []

for i in range(3):
    y_true = actual[i]
    y_pred = preds[i]

    # Calculate evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    rec = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    results.append([
        names[i],
        acc * 100,
        prec * 100,
        rec * 100,
        f1 * 100
    ])

    # Display task evaluation
    print(f"\n{names[i]} Evaluation")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    # Calculate and display confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred
    )
    print("Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        names[i] + " Confusion Matrix"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# 12. Display Task-wise Results Table
results_df = pd.DataFrame(
    results,
    columns=[
        "Task",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)
print("\nTask-wise Evaluation:")
print(
    results_df.round(2).to_string(
        index=False
    )
)

# 13. Predict Outcomes for One Test Student
sample = X_test[0:1]
model.eval()

with torch.no_grad():
    sample_outputs = model(sample)

    sample_probs = [
        torch.sigmoid(output).item()
        for output in sample_outputs
    ]

# Prediction labels for each task
labels = [
    ["FAIL", "PASS"],
    ["POOR ATTENDANCE", "GOOD ATTENDANCE"],
    ["HIGH ACADEMIC RISK", "LOW ACADEMIC RISK"]
]
print("\nSample Student Predictions:")

for i in range(3):
    prediction = int(
        sample_probs[i] >= 0.5
    )

    print(
        f"{names[i]}: "
        f"Probability = {sample_probs[i]:.4f}, "
        f"Prediction = {labels[i][prediction]}"
    )
