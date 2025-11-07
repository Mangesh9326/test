import os
import numpy as np
import hashlib
from typing import List, Set

# === Config ===
FOLDER = "british-fiction-corpus"
K = 5  # shingle size
NUM_HASHES = 100


def get_files(folder: str) -> List[str]:
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]


def shingle(text: str, k: int) -> Set[str]:
    return set(text[i:i+k] for i in range(len(text) - k + 1))


def hash_family(i: int):
    def hash_fn(x: str):
        return int(hashlib.sha256((str(i) + x).encode('utf-8')).hexdigest(), 16)
    return hash_fn


def minhash_signature(shingle_set: Set[str], hash_funcs: List) -> List[int]:
    signature = []
    for h in hash_funcs:
        min_hash = min(h(shingle) for shingle in shingle_set)
        signature.append(min_hash)
    return signature


def main():
    file_paths = get_files(FOLDER)
    print(f"Found {len(file_paths)} text files.")

    # Step 1: Read files and shingle them
    documents = []
    all_shingles = set()

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().lower()
            shingle_set = shingle(text, K)
            documents.append(shingle_set)
            all_shingles.update(shingle_set)

    print(f"Total unique shingles: {len(all_shingles)}")

    # Step 2: Create hash functions
    hash_funcs = [hash_family(i) for i in range(NUM_HASHES)]

    # Step 3: Compute MinHash signatures
    signatures = []
    for doc_shingles in documents:
        sig = minhash_signature(doc_shingles, hash_funcs)
        signatures.append(sig)

    print(f"Generated MinHash signatures for {len(signatures)} documents.")

    # Example: Show pairwise similarity (Jaccard estimate)
    def compute_similarity(sig1, sig2):
        return np.mean(np.array(sig1) == np.array(sig2))

    print("\nEstimated Jaccard Similarity (using MinHash):")
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            sim = compute_similarity(signatures[i], signatures[j])
            print(f"Doc {i} vs Doc {j}: {sim:.3f}")


if __name__ == "__main__":
    main()
