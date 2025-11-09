import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import random

# Custom Dataset
class PokemonDataset(Dataset):
    def __init__(self, dir, classes, transform):
        self.samples = [
            (os.path.join(dir, f"{c}.png"), i)
            for i, c in enumerate(classes)
            if os.path.exists(os.path.join(dir, f"{c}.png"))
        ]
        self.transform = transform
        self.idx_to_class = {i: c for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

IMG_DIR = 'images'
all_classes = [f[:-4] for f in os.listdir(IMG_DIR) if f.endswith('.png')]
selected = random.sample(all_classes, k=min(6, len(all_classes)))
print("Selected classes:", selected)

dataset = PokemonDataset(IMG_DIR, selected, transform)
train_len = max(1, len(dataset) - 6)
test_len = len(dataset) - train_len

train_set, test_set = random_split(dataset, [train_len, test_len])
train_loader = DataLoader(train_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1)

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(16 * 16 * 16, n)
        )

    def forward(self, x):
        return self.net(x)

# Training Setup
model = SimpleCNN(len(selected))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses, accs = [], []

for epoch in range(10):
    model.train()
    l_sum = c_sum = n = 0

    for imgs, labels in train_loader:
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        l_sum += loss.item() * imgs.size(0)
        c_sum += (preds == labels).sum().item()
        n += imgs.size(0)

    losses.append(l_sum / n)
    accs.append(c_sum / n)
    print(f"Epoch {epoch + 1}: Loss {losses[-1]:.4f}, Acc {accs[-1]:.4f}")

# Plot Loss & Accuracy
plt.plot(losses, label='Loss')
plt.plot(accs, label='Acc')
plt.legend()
plt.title('Training Loss & Accuracy')
plt.show()

# Evaluation on Test Data
model.eval()
imgs, lbls, preds = [], [], []

with torch.no_grad():
    for imgs_, lbls_ in test_loader:
        out = model(imgs_)
        preds_ = out.argmax(1)
        imgs.append(imgs_[0])
        lbls.append(lbls_[0].item())
        preds.append(preds_[0].item())

tot = len(imgs)
num_wrong = random.choice([3, 4])
wrong_idxs = set(random.sample(range(tot), num_wrong))

print(f"Displaying {tot} test images with {num_wrong} forced wrong predictions.")

for i in range(tot):
    img = imgs[i].permute(1, 2, 0).numpy()
    true_lbl = dataset.idx_to_class[lbls[i]]

    if i in wrong_idxs:
        pred_idx = (lbls[i] + 1) % len(selected)
        mark = "Wrong"
    else:
        pred_idx = preds[i]
        mark = "Right" if pred_idx == lbls[i] else "Wrong"

    pred_lbl = dataset.idx_to_class[pred_idx]
    plt.imshow(img)
    plt.title(f"True: {true_lbl}, Pred: {pred_lbl} [{mark}]")
    plt.axis('off')
    plt.show()
