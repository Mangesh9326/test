import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence

max_features = 10000  # Use the top 10,000 words
maxlen = 500          # Maximum review length

print("Loading data...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features
)
print("Training sequences:", len(x_train))
print("Testing sequences:", len(x_test))

# 3. PAD SEQUENCES
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print("Data padded successfully.")
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# 4. BUILD RNN MODEL
model = Sequential([
    Embedding(max_features, 32),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])

# 5. COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 6. DISPLAY MODEL ARCHITECTURE
model.summary()

# 7. TRAIN MODEL
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# 8. PLOT ACCURACY GRAPH
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("RNN Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 9. PLOT LOSS GRAPH
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("RNN Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# 10. EVALUATE MODEL
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# 11. PREDICT SENTIMENT OF ONE REVIEW
# Get the word index dictionary
word_index = imdb.get_word_index()

# Reverse the dictionary: ID -> word
reverse_word_index = {
    value + 3: key
    for key, value in word_index.items()
}

# Decode the first test review
review = x_test[0]

review_text = " ".join([
    reverse_word_index.get(index, "?")
    for index in review
    if index != 0
])
print("Actual Review:")
print(review_text)

# Predict sentiment
prediction = model.predict(
    x_test[:1],
    verbose=0
)[0][0]

print(
    "\nPositive sentiment probability:",
    round(float(prediction), 4)
)

if prediction >= 0.5:
    print("Predicted Sentiment: Positive")
else:
    print("Predicted Sentiment: Negative")

print(
    "Actual Sentiment:",
    "Positive" if y_test[0] == 1 else "Negative"
)
