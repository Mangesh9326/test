import json
import random
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


# 1. Configuration

SEED = 42
IMAGE_HEIGHT = 48
MAX_IMAGE_WIDTH = 1024
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 128
NUM_WORKERS = 0
PATIENCE = 3

MODEL_PATH = Path("cnn_handwriting_model.pth")
VOCAB_PATH = Path("char_to_idx.json")


# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Limit CPU threads to reduce overhead
torch.set_num_threads(4)

# Select the computation device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# 2. Load IAM line dataset
dataset = load_dataset("Teklia/IAM-line")

print(dataset)
print("Example transcription:", dataset["train"][0]["text"])



# 3. Build vocabulary from training data only

vocab = sorted(set("".join(dataset["train"]["text"])))

# CTC blank must have index 0
char_to_idx = {
    char: idx + 1
    for idx, char in enumerate(vocab)
}
char_to_idx["<BLANK>"] = 0

idx_to_char = {
    idx: char
    for char, idx in char_to_idx.items()
}

num_classes = len(char_to_idx)

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(char_to_idx, f, ensure_ascii=False, indent=2)

print("Vocabulary size:", num_classes)


# Check for characters missing from training vocabulary
for split in ["validation", "test"]:
    unseen = set(
        "".join(dataset[split]["text"])
    ) - set(vocab)

    if unseen:
        print(
            f"Warning: unseen characters in {split}: "
            f"{sorted(unseen)}"
        )


# 4. Image preprocessing

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def resize_image(image):
    image = image.convert("L")

    # Correct image orientation if EXIF metadata is present
    image = ImageOps.exif_transpose(image)

    # Preserve the original aspect ratio
    scale = IMAGE_HEIGHT / image.height
    new_width = max(
        1,
        round(image.width * scale)
    )

    # Prevent excessively large images from exhausting memory
    if new_width > MAX_IMAGE_WIDTH:
        new_width = MAX_IMAGE_WIDTH

    image = image.resize(
        (new_width, IMAGE_HEIGHT),
        Image.Resampling.LANCZOS
    )

    return image


# 5. Dataset

class IAMLineDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = resize_image(item["image"])

        width = image.width
        image_tensor = transform(image)

        # Do not silently drop unknown characters.
        text = item["text"]

        unknown = set(text) - set(vocab)
        if unknown:
            raise ValueError(
                f"Unknown characters {unknown} in: {text!r}"
            )

        target = torch.tensor(
            [char_to_idx[ch] for ch in text],
            dtype=torch.long
        )

        return image_tensor, target, width, text


# 6. Variable-width batch collation

def collate_fn(batch):
    images, targets, widths, texts = zip(*batch)

    # CNN reduces width by a factor of 4.
    # Pad input widths to a multiple of 4.
    max_width = max(widths)
    max_width = ((max_width + 3) // 4) * 4

    padded_images = []

    for image in images:
        padding = max_width - image.shape[-1]

        # Normalized white background is +1.
        image = F.pad(
            image,
            (0, padding, 0, 0),
            value=1.0
        )
        padded_images.append(image)

    images = torch.stack(padded_images)
    targets = torch.cat(targets)

    target_lengths = torch.tensor(
        [len(t) for t in batch_targets(batch)],
        dtype=torch.long
    )

    widths = torch.tensor(widths, dtype=torch.long)

    return images, targets, target_lengths, widths, texts


def batch_targets(batch):
    return [item[1] for item in batch]


# 7. DataLoaders

train_dataset = IAMLineDataset(dataset["train"])
val_dataset = IAMLineDataset(dataset["validation"])
test_dataset = IAMLineDataset(dataset["test"])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)


# 8. CNN + BiLSTM + CTC model

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))
        )

        # Height after pooling: 64 -> 32 -> 16 -> 8 -> 4
        self.map_to_seq = nn.Linear(128 * 3, 128)

        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x)

        # [B, C, H, W] -> [B, W, C*H]
        sequence = features.permute(
            0, 3, 1, 2
        ).contiguous().flatten(start_dim=2)

        sequence = F.relu(self.map_to_seq(sequence))
        sequence, _ = self.rnn(sequence)
        sequence = self.dropout(sequence)

        return self.fc(sequence)


model = CRNN(
    num_classes=num_classes,
    hidden_size=HIDDEN_SIZE
).to(device)

criterion = nn.CTCLoss(
    blank=0,
    zero_infinity=True
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

print(model)


# 9. CTC decoding and Character Error Rate

def decode_prediction(logits):
    # logits: [time, classes]
    indices = logits.argmax(dim=-1).tolist()

    decoded = []
    previous = None

    for idx in indices:
        # Collapse repeated tokens before removing blanks
        if idx != previous and idx != 0:
            decoded.append(idx_to_char[idx])

        previous = idx

    return "".join(decoded)


def edit_distance(reference, hypothesis):
    previous_row = list(range(len(hypothesis) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current_row = [i]

        for j, hyp_char in enumerate(hypothesis, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = (
                previous_row[j - 1]
                + (ref_char != hyp_char)
            )

            current_row.append(
                min(insertion, deletion, substitution)
            )

        previous_row = current_row

    return previous_row[-1]


def calculate_cer(reference, hypothesis):
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    return edit_distance(reference, hypothesis) / len(reference)


# 10. Train and evaluate one epoch

def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    total_edits = 0
    total_characters = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, targets, target_lengths, widths, texts in loader:
            images = images.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            # [B, T, C] -> [T, B, C]
            log_probs = F.log_softmax(
                logits, dim=2
            ).permute(1, 0, 2)

            # Two horizontal pooling layers, each with stride 2
            input_lengths = (widths // 4).to(torch.long)

            if torch.any(input_lengths < target_lengths):
                raise ValueError(
                    "An image has too few CTC time steps for its label. "
                    "Increase MAX_IMAGE_WIDTH or adjust preprocessing."
                )

            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

            if not torch.isfinite(loss):
                print("Warning: non-finite loss; batch skipped.")
                continue

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Greedy CTC decoding for CER
            predictions = logits.detach().argmax(dim=2)

            for i in range(batch_size):
                pred = decode_prediction(logits[i].detach())
                reference = texts[i]

                total_edits += edit_distance(reference, pred)
                total_characters += len(reference)

    average_loss = total_loss / max(total_samples, 1)
    cer = total_edits / max(total_characters, 1)

    return average_loss, cer


# 11. Training with validation and early stopping

best_val_cer = float("inf")
epochs_without_improvement = 0
history = {
    "train_loss": [],
    "val_loss": [],
    "val_cer": []
}

start_time = time.time()

for epoch in range(EPOCHS):
    train_loss, train_cer = run_epoch(
        train_loader, training=True
    )

    val_loss, val_cer = run_epoch(
        val_loader, training=False
    )

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_cer"].append(val_cer)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train CER: {train_cer:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val CER: {val_cer:.4f} | "
        f"LR: {current_lr:.6f}"
    )

    # Select the checkpoint with the best validation CER
    if np.isfinite(val_cer) and val_cer < best_val_cer:
        best_val_cer = val_cer
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "char_to_idx": char_to_idx,
                "num_classes": num_classes,
                "image_height": IMAGE_HEIGHT,
                "max_image_width": MAX_IMAGE_WIDTH,
                "hidden_size": HIDDEN_SIZE,
                "best_val_cer": best_val_cer,
                "epoch": epoch + 1
            },
            MODEL_PATH
        )

        print("  Best model checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation CER did not improve.")
        break

print(f"Training time: {(time.time() - start_time) / 60:.1f} minutes")


# 12. Load best model and evaluate on held-out test set

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_cer = run_epoch(
    test_loader, training=False
)

print("\n========== FINAL TEST RESULTS ==========")
print("Best epoch:", checkpoint["epoch"])
print(f"Best validation CER: {checkpoint['best_val_cer']:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test CER: {test_cer:.4f}")
print(f"Test character accuracy estimate: {max(0.0, 1-test_cer)*100:.2f}%")
print("Model saved to:", MODEL_PATH)


# 13. Plot learning curves

epochs_ran = range(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(10, 4))
plt.plot(epochs_ran, history["train_loss"], label="Train loss")
plt.plot(epochs_ran, history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("CTC loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(epochs_ran, history["val_cer"], label="Validation CER")
plt.xlabel("Epoch")
plt.ylabel("Character Error Rate")
plt.title("Validation Character Error Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("validation_cer.png", dpi=150)
plt.show()


# 14. Display a prediction from the TEST set

model.eval()

image, target, width, actual_text = test_dataset[0]

with torch.no_grad():
    logits = model(
        image.unsqueeze(0).to(device)
    )[0]

    predicted_text = decode_prediction(logits.cpu())

display_image = (
    image.squeeze(0).cpu() * 0.5 + 0.5
).clamp(0, 1).numpy()

# Remove padding from the displayed image
display_image = display_image[:, :width]

plt.figure(figsize=(16, 4))
plt.imshow(display_image, cmap="gray")
plt.axis("off")
plt.title("IAM Test Image")
plt.tight_layout()
plt.show()

print("\n========== SAMPLE PREDICTION ==========")
print("Actual text:   ", actual_text)
print("Predicted text:", predicted_text)
print(f"Sample CER:    {calculate_cer(actual_text, predicted_text):.4f}")
