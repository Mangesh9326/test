# 5.3 Identify the Most Frequently Repeated Amino Acid

def read_sequence(file_path):
    # Read a protein sequence from a text file
    with open(file_path, "r") as file:
        return file.read().strip().upper()


def count_amino_acids(sequence):
    # Count occurrences of standard amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    counts = {}

    print(f"Protein Sequence: {sequence}")
    print(f"Sequence Length: {len(sequence)} amino acids")

    print("\nAMINO ACID FREQUENCY")
    print("-" * 30)

    for aa in amino_acids:
        count = sequence.count(aa)
        counts[aa] = count
        if count > 0:
            print(f"{aa} : {count}")

    # Identify the amino acid with the highest frequency
    if not sequence:
        print("\nThe sequence is empty.")
        return

    highest_count = max(counts.values())

    if highest_count == 0:
        print("\nNo standard amino acids were found.")
        return

    most_frequent = [
        aa for aa, count in counts.items()
        if count == highest_count
    ]

    print("\nMOST FREQUENT AMINO ACID")
    print("-" * 30)
    print(f"Amino acid(s): {', '.join(most_frequent)}")
    print(f"Occurrence(s): {highest_count}")


def main():
    # Load the protein sequence and analyze amino acid frequencies
    try:
        sequence = read_sequence("seq1.txt")
        count_amino_acids(sequence)

    except FileNotFoundError:
        print("Error: seq1.txt was not found.")
        print("Place seq1.txt in the same folder as this script.")

    except OSError as e:
        print(f"Error reading the sequence file: {e}")


if __name__ == "__main__":
    main()