# 5.1 Identify the Unmatched Amino Acids in Two Sequences

def read_sequence(file_path):
    # Read a protein sequence from a text file
    with open(file_path, "r") as file:
        return file.read().strip().upper()


def compare_sequences(seq1, seq2):
    # Compare amino acids at corresponding positions
    max_len = max(len(seq1), len(seq2))
    differences = []

    for i in range(max_len):
        aa1 = seq1[i] if i < len(seq1) else "-"
        aa2 = seq2[i] if i < len(seq2) else "-"

        if aa1 != aa2:
            differences.append((i + 1, aa1, aa2))

    # Display unmatched amino acids
    print("\n" + "=" * 45)
    print("SEQUENCE COMPARISON RESULTS")
    print("=" * 45)

    if differences:
        print(f"Total unmatched positions: {len(differences)}")
        print("\nPosition    Sequence 1    Sequence 2")
        print("-" * 40)

        for position, aa1, aa2 in differences:
            print(f"{position:<12}{aa1:<14}{aa2}")
    else:
        print("No unmatched amino acids were found.")

    return differences


def main():
    # Load the two protein sequences
    try:
        seq1 = read_sequence("seq1.txt")
        seq2 = read_sequence("seq2.txt")

        print(f"Sequence 1 ({len(seq1)} amino acids): {seq1}")
        print(f"Sequence 2 ({len(seq2)} amino acids): {seq2}")

        compare_sequences(seq1, seq2)

    except FileNotFoundError as e:
        print(f"Error: File '{e.filename}' was not found.")
        print("Place seq1.txt and seq2.txt in the same folder as this script.")

    except OSError as e:
        print(f"Error reading sequence files: {e}")


if __name__ == "__main__":
    main()
