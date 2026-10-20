import torch
import torch.nn as nn
import torch.optim as optim

# 1. DEFINE RECURSIVE NEURAL NETWORK
class RecursiveNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        # Convert word IDs into vectors
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Combine left and right child representations
        self.composition = nn.Linear(hidden_dim * 2, hidden_dim)

        # Classify the final tree representation
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, tree):

        # If the node is a leaf (word ID)
        if isinstance(tree, int):
            word = torch.tensor(tree, dtype=torch.long)
            return self.embedding(word)

        # Recursively process left and right children
        left_vector = self.forward(tree[0])
        right_vector = self.forward(tree[1])

        # Combine both child vectors
        combined = torch.cat((left_vector, right_vector))

        # Generate parent representation
        parent_vector = torch.tanh(
            self.composition(combined)
        )
        return parent_vector

# 2. CREATE SAMPLE TREE DATA
# Label 1 = Positive sentiment
# Label 0 = Negative sentiment
training_data = [
    (((10, 25), (4, 82)), 1),  # Positive: "the movie was great"
    (((10, 25), (4, 50)), 0),  # Negative: "the movie was terrible"
    ((82, 25), 1),             # Positive: "great movie"
    ((50, 25), 0)              # Negative: "terrible movie"
]

# 3. INITIALIZE MODEL
model = RecursiveNN(
    vocab_size=100,
    embed_dim=16,
    hidden_dim=16,
    num_classes=2
)

# 4. DEFINE LOSS AND OPTIMIZER
optimizer = optim.Adam(
    model.parameters(),
    lr=0.05
)
criterion = nn.CrossEntropyLoss()

# 5. TRAIN THE MODEL
print("Starting training...")
for epoch in range(20):
    total_loss = 0

    for tree, label in training_data:
        optimizer.zero_grad()

        # Generate representation of the complete tree
        root_vector = model(tree)

        # Predict sentiment from the root vector
        logits = model.classifier(root_vector).unsqueeze(0)

        # Actual label
        target = torch.tensor([label], dtype=torch.long)

        # Calculate loss and update parameters
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1:02d}/20 | "
            f"Loss: {total_loss:.4f}"
        )

# 6. TEST THE TRAINED MODEL
print("\nTesting trained model:")
model.eval()

with torch.no_grad():
    for tree, label in training_data:

        root_vector = model(tree)
        logits = model.classifier(root_vector)

        prediction = torch.argmax(logits, dim=-1).item()

        actual = "Positive" if label == 1 else "Negative"
        predicted = "Positive" if prediction == 1 else "Negative"

        print("Tree:", tree)
        print("Actual sentiment:", actual)
        print("Predicted sentiment:", predicted)
        print("-" * 40)
