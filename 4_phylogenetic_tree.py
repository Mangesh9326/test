
# PHYLOGENETIC TREE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# Define species and their DNA sequences
data = {
    "Human": "ATCGGCTA",
    "Chimp": "ATCGGCTA",
    "Monkey": "ATCGACTA",
    "Mouse": "TTAGACTG"
}

species_names = list(data.keys())
sequences = list(data.values())
n = len(species_names)

# Check that all sequences have the same length
if len({len(seq) for seq in sequences}) != 1:
    raise ValueError("All sequences must have the same length.")

# Calculate percentage similarity and evolutionary distance
def calculate_metrics(seq1, seq2):
    matches = sum(a == b for a, b in zip(seq1, seq2))
    similarity = matches / len(seq1)
    distance = 1 - similarity
    return similarity * 100, distance

# Create similarity and distance matrices
similarity_matrix = np.zeros((n, n))
distance_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity, distance = calculate_metrics(sequences[i], sequences[j])
        similarity_matrix[i, j] = similarity
        distance_matrix[i, j] = distance

df_similarity = pd.DataFrame(
    similarity_matrix, index=species_names, columns=species_names
)
df_distance = pd.DataFrame(
    distance_matrix, index=species_names, columns=species_names
)

print("=" * 60)
print("PHYLOGENETIC TREE CONSTRUCTION")
print("=" * 60)

print("\nDNA Sequences:")
for species, sequence in data.items():
    print(f"{species:<10}: {sequence}")

print("\nSequence Similarity Matrix (%):")
print(df_similarity.round(2).to_string())

print("\nEvolutionary Distance Matrix:")
print(df_distance.round(3).to_string())

# Identify the pair with the smallest distance
upper_triangle = np.triu_indices(n, k=1)
closest_index = np.argmin(distance_matrix[upper_triangle])
i = upper_triangle[0][closest_index]
j = upper_triangle[1][closest_index]

print("\nClosest Species Pair:")
print(f"{species_names[i]} and {species_names[j]}")
print(f"Distance: {distance_matrix[i, j]:.3f}")
print(f"Similarity: {similarity_matrix[i, j]:.2f}%")

# Convert the distance matrix into condensed form for clustering
condensed_distance = squareform(distance_matrix)

# Construct the hierarchical tree using average linkage
linked = linkage(condensed_distance, method="average")

print("\nClustering Method: Average Linkage")

# Plot the phylogenetic dendrogram
plt.figure(figsize=(9, 5))
dendrogram(
    linked,
    labels=species_names,
    orientation="right",
    distance_sort="ascending"
)

plt.title("Phylogenetic Tree of Four Species")
plt.xlabel("Evolutionary Distance")
plt.ylabel("Species")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()