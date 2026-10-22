# 5.2 Calculate the Percentage of Pattern Matching

def read_sequence(file_path):
    # Read a protein sequence from a text file
    with open(file_path, "r") as file:
        return file.read().strip().upper()


def calculate_match_percentage(seq1, seq2):
    # Compare corresponding positions in both sequences
    max_len = max(len(seq1), len(seq2))

    if max_len == 0:
        return 0.0, 0, 0

    matches = 0

    for i in range(max_len):
        aa1 = seq1[i] if i < len(seq1) else "-"
        aa2 = seq2[i] if i < len(seq2) else "-"

        if aa1 == aa2:
            matches += 1

    # Calculate the percentage of matching positions
    percentage = (matches / max_len) * 100
    return percentage, matches, max_len


def main():
    # Read the two sequences and display the comparison results
    try:
        seq1 = read_sequence("seq1.txt")
        seq2 = read_sequence("seq2.txt")
        
        print(f"Sequence 1 ({len(seq1)} amino acids): {seq1}")
        print(f"Sequence 2 ({len(seq2)} amino acids): {seq2}")

        percentage, matches, total = calculate_match_percentage(seq1, seq2)

        print("\nMATCHING RESULTS")
        print("-" * 45)
        print(f"Matching positions : {matches}")
        print(f"Total positions    : {total}")
        print(f"Matching percentage: {percentage:.2f}%")

    except FileNotFoundError as e:
        print(f"Error: File '{e.filename}' was not found.")
        print("Ensure seq1.txt and seq2.txt are in the same folder.")

    except OSError as e:
        print(f"Error reading sequence files: {e}")


if __name__ == "__main__":
    main()