import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)

from torch.utils.data import TensorDataset, DataLoader

# STEP 1: DEVICE AND REPRODUCIBILITY
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device, flush=True)

torch.manual_seed(42)
np.random.seed(42)

# STEP 2: LOAD SMALL DATASET
digits = load_digits()

X_np = digits.images.astype(np.float32) / 16.0
y = digits.target  # Used ONLY for evaluation

# Shape: (samples, channels, height, width)
X = torch.tensor(X_np).unsqueeze(1)

print("Dataset shape:", X.shape, flush=True)
print("Number of images:", len(X), flush=True)

# STEP 3: CNN FEATURE EXTRACTOR
class FeatureCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.features(x)

# STEP 4: PRETEXT EXEMPLAR LEARNING
print(
    "\n--- PRETEXT EXEMPLAR LEARNING ---",
    flush=True
)

NUM_EXEMPLARS = 100
VIEWS_PER_EXEMPLAR = 5

# Select 100 images as exemplars
exemplars = X[:NUM_EXEMPLARS]

augmented_images = []
exemplar_labels = []

for i in range(NUM_EXEMPLARS):
    for j in range(VIEWS_PER_EXEMPLAR):

        image = exemplars[i].clone()

        # Create a slightly different view
        if j > 0:
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)

        augmented_images.append(image)
        exemplar_labels.append(i)

augmented_images = torch.stack(augmented_images)

exemplar_labels = torch.tensor(
    exemplar_labels,
    dtype=torch.long
)

exemplar_dataset = TensorDataset(
    augmented_images,
    exemplar_labels
)

exemplar_loader = DataLoader(
    exemplar_dataset,
    batch_size=32,
    shuffle=True
)

print(
    "Exemplar training samples:",
    len(exemplar_dataset),
    flush=True
)

# STEP 5: DEFINE EXEMPLAR CNN
class ExemplarCNN(nn.Module):

    def __init__(self, num_exemplars):
        super().__init__()

        self.feature_extractor = FeatureCNN()
        self.classifier = nn.Linear(64, num_exemplars)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# INITIALIZE EXEMPLAR MODEL
exemplar_model = ExemplarCNN(NUM_EXEMPLARS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    exemplar_model.parameters(),
    lr=0.001
)

exemplar_losses = []

# TRAIN EXEMPLAR CNN
for epoch in range(5):

    exemplar_model.train()
    total_loss = 0

    for images, labels in exemplar_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = exemplar_model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(exemplar_loader)
    exemplar_losses.append(avg_loss)

    print(
        f"Exemplar Epoch {epoch + 1}/5 | "
        f"Loss: {avg_loss:.4f}",
        flush=True
    )
print("Exemplar learning completed!", flush=True)

# STEP 6: CLUSTERING
print("\n--- CLUSTERING ---", flush=True)

# EXTRACT FEATURES
exemplar_model.eval()

feature_loader = DataLoader(
    TensorDataset(X),
    batch_size=128,
    shuffle=False
)

feature_batches = []

with torch.no_grad():
    for (images,) in feature_loader:

        images = images.to(device)

        features_batch = (
            exemplar_model.feature_extractor(images)
        )

        feature_batches.append(
            features_batch.cpu().numpy()
        )

features = np.concatenate(feature_batches, axis=0)
print(
    "Extracted feature shape:",
    features.shape,
    flush=True
)

# APPLY K-MEANS CLUSTERING
NUM_CLUSTERS = 10
kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(features)
print("KMeans clustering completed!", flush=True)

# EVALUATE CLUSTER QUALITY
# Actual labels are NOT used to train either model.
ari = adjusted_rand_score(y, cluster_labels)

nmi = normalized_mutual_info_score(
    y,
    cluster_labels
)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# STEP 7: TRAIN CNN TO PREDICT CLUSTER IDS
cluster_dataset = TensorDataset(
    X,
    torch.tensor(cluster_labels, dtype=torch.long)
)

cluster_loader = DataLoader(
    cluster_dataset,
    batch_size=64,
    shuffle=True
)

# DEFINE CLUSTER CNN
class ClusterCNN(nn.Module):

    def __init__(self, num_clusters):
        super().__init__()

        self.feature_extractor = FeatureCNN()
        self.classifier = nn.Linear(64, num_clusters)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# INITIALIZE CLUSTER MODEL
cluster_model = ClusterCNN(NUM_CLUSTERS).to(device)
cluster_optimizer = optim.Adam(
    cluster_model.parameters(),
    lr=0.001
)
cluster_losses = []

# TRAIN CLUSTER CNN
for epoch in range(5):
    cluster_model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in cluster_loader:

        images = images.to(device)
        labels = labels.to(device)

        cluster_optimizer.zero_grad()

        outputs = cluster_model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        cluster_optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    avg_loss = total_loss / len(cluster_loader)
    accuracy = 100 * correct / total

    cluster_losses.append(avg_loss)

    print(
        f"Cluster Epoch {epoch + 1}/5 | "
        f"Loss: {avg_loss:.4f} | "
        f"Cluster-label accuracy: {accuracy:.2f}%",
        flush=True
    )
print("Cluster CNN training completed!", flush=True)

# STEP 8: SHOW SAMPLE IMAGE AND PREDICTION
cluster_model.eval()

sample_index = 7
sample_image = X[sample_index].unsqueeze(0).to(device)

with torch.no_grad():
    output = cluster_model(sample_image)

    predicted_cluster = output.argmax(dim=1).item()

print("\nSample prediction")
print(
    "Actual digit (evaluation only):",
    y[sample_index]
)

print(
    "KMeans cluster:",
    cluster_labels[sample_index]
)

print(
    "CNN predicted cluster:",
    predicted_cluster
)

# DISPLAY SAMPLE IMAGE
plt.figure(figsize=(4, 4))
plt.imshow(
    X_np[sample_index],
    cmap="gray"
)

plt.title(
    f"Actual Digit: {y[sample_index]}\n"
    f"Predicted Cluster: {predicted_cluster}"
)

plt.axis("off")
plt.show()

# STEP 9: CLUSTER COMPOSITION
print("\nMost common actual digits in each cluster")
print(
    "(Labels are inspected only after clustering.)"
)

for cluster_id in range(NUM_CLUSTERS):

    indices = np.where(
        cluster_labels == cluster_id
    )[0]

    counts = np.bincount(
        y[indices],
        minlength=10
    )

    top_digits = np.argsort(counts)[::-1][:3]

    print(
        f"Cluster {cluster_id}:",
        [
            (int(d), int(counts[d]))
            for d in top_digits
            if counts[d] > 0
        ]
    )

# STEP 10: PLOT LOSS GRAPHS
plt.figure(figsize=(7, 4))
plt.plot(
    range(1, len(exemplar_losses) + 1),
    exemplar_losses,
    marker="o",
    label="Exemplar Loss"
)

plt.plot(
    range(1, len(cluster_losses) + 1),
    cluster_losses,
    marker="o",
    label="Cluster CNN Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Unsupervised Feature Learning")
plt.legend()
plt.grid(True)
plt.show()