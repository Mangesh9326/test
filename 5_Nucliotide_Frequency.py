# 5.4 Count the Number of Repetitions of Each Nucleotide

def count_nucleotides(sequence):
    # Convert the DNA sequence to uppercase
    sequence = sequence.upper()

    # Count the four DNA nucleotides
    nucleotides = ["A", "C", "G", "T"]

    print("=" * 40)
    print("PRACTICAL 5.4: NUCLEOTIDE FREQUENCY")
    print("=" * 40)
    print(f"DNA Sequence: {sequence}")
    print(f"Sequence Length: {len(sequence)}")

    print("\nNUCLEOTIDE COUNTS")
    print("-" * 30)

    for nucleotide in nucleotides:
        count = sequence.count(nucleotide)
        print(f"{nucleotide} : {count}")

    # Check whether the sequence contains valid DNA nucleotides
    if not any(base in nucleotides for base in sequence):
        print("\nNo valid DNA nucleotides were found.")

    # Report any characters outside the standard DNA alphabet
    invalid = sorted(set(sequence) - set(nucleotides))
    if invalid:
        print(f"\nInvalid or unexpected characters: {', '.join(invalid)}")


def main():
    # Define the DNA sequence for analysis
    sequence = "ACTGGACTTTAC"

    # Count and display nucleotide frequencies
    count_nucleotides(sequence)


if __name__ == "__main__":
    main()