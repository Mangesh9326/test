import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# 1. GENERATE SYNTHETIC DATA
X_np, _ = make_blobs(
    n_samples=1000,
    n_features=10,
    centers=3,
    random_state=42
)

X_tensor = torch.tensor(X_np, dtype=torch.float32)
print("Original data shape:", X_np.shape)

# 2. APPROACH A: PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_np)

print("\nPCA Output Shape:", X_pca.shape)
print(
    "PCA Explained Variance Ratio:",
    pca.explained_variance_ratio_
)

print(
    "Total Explained Variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)

# 3. DEFINE LINEAR AUTOENCODER
class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder reduces dimensions
        self.encoder = nn.Linear(
            input_dim,
            latent_dim,
            bias=False
        )

        # Decoder reconstructs original dimensions
        self.decoder = nn.Linear(
            latent_dim,
            input_dim,
            bias=True
        )

    def forward(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        return x_recon, h

# 4. SET HYPERPARAMETERS
input_dim = 10
latent_dim = 3

model = LinearAutoencoder(
    input_dim,
    latent_dim
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)

# 5. TRAIN THE AUTOENCODER
model.train()
for epoch in range(200):
    optimizer.zero_grad()

    reconstructions, latent_factors = model(X_tensor)
    loss = criterion(
        reconstructions,
        X_tensor
    )

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch [{epoch + 1}/200], "
            f"Reconstruction MSE Loss: {loss.item():.4f}"
        )

# 6. EXTRACT LEARNED FACTORS
model.eval()
with torch.no_grad():
    reconstructions, latent_factors = model(X_tensor)
    latent_factors = latent_factors.numpy()
    reconstructions = reconstructions.numpy()
    W = model.decoder.weight.detach().numpy()

print(
    "\nAutoencoder Latent Shape:",
    latent_factors.shape
)

print(
    "Factor Loading Matrix Shape:",
    W.shape
)

# 7. COMPARE RECONSTRUCTION ERROR
pca_reconstructed = pca.inverse_transform(X_pca)
pca_mse = np.mean(
    (X_np - pca_reconstructed) ** 2
)

ae_mse = np.mean(
    (X_np - reconstructions) ** 2
)

print(
    "\nPCA Reconstruction MSE:",
    round(pca_mse, 4)
)

print(
    "Autoencoder Reconstruction MSE:",
    round(ae_mse, 4)
)

# 8. PLOT PCA EXPLAINED VARIANCE
plt.figure(figsize=(7, 4))
plt.bar(
    range(1, 4),
    pca.explained_variance_ratio_
)

plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.xticks([1, 2, 3])
plt.grid(axis="y")
plt.show()

# 9. PLOT RECONSTRUCTION ERROR COMPARISON
plt.figure(figsize=(7, 4))
plt.bar(
    ["PCA", "Linear Autoencoder"],
    [pca_mse, ae_mse]
)

plt.title("Reconstruction Error Comparison")
plt.ylabel("Mean Squared Error")
plt.grid(axis="y")
plt.show()
