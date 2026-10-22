# BIOINFORMATICS: SEQUENCE MANIPULATION USING BIOPYTHON
# Read and parse sequence data from files
# Perform basic sequence manipulations (e.g., reverse complement, translation)

import random
from Bio.Seq import Seq

# 1. DNA SEQUENCE: COMPLEMENT AND REVERSE COMPLEMENT

dna = Seq(
    "AATTCTAGGCCGTCCTGGCGATAAAGGTGAGCCAGGTAAACATGTGCCCGGAGTTGCACCACCAGGACCA"
    "CCAGGTCGACCGGGTGATATGGGACCACCTGGACCACCTGGGCCACCAGGCGAGAAAGGAAAACCTGGAA"
    "CAAAAGGCCCAATAGGACCACCGGGAGACCAAGGAAATCCAGGTATTTATATTCGTAGTTAATGCTTGAT"
    "TCGAGAATT"
)

print("Original DNA:", dna)
print("Complement:", dna.complement())
print("Reverse complement:", dna.reverse_complement())


# 2. TRANSCRIPTION: DNA TO RNA

rna = dna.transcribe()

print("\nRNA:", rna)


# 3. TRANSLATION: DNA AND RNA TO PROTEIN

print("DNA to protein:", dna.translate())
print("RNA to protein:", rna.translate())



# 4. GENERATE A RANDOM SEQUENCE

def generate_random_sequence(sequence_length, probability_distribution):
    """Generate a random DNA sequence using given probabilities."""

    sequence = ""

    for i in range(sequence_length):
        sequence += random.choices(
            ["A", "C", "G", "T"],
            weights=probability_distribution,
            k=1
        )[0]

    return sequence


our_random_seq = generate_random_sequence(
    100, [0.25, 0.25, 0.25, 0.25]
)

print("\nRandom DNA sequence:", our_random_seq)
print("Sequence length:", len(our_random_seq))


# 5. SEQUENCE SLICING AND LENGTH

my_dna = Seq("ATCGATGATAGGATA")
my_protein = Seq("KGLYQRGD")

a_sub_sequence = my_dna[3:7]
the_sequence_length = len(my_dna)

print("\nDNA sequence:", my_dna)
print("DNA length:", the_sequence_length)
print("Subsequence:", a_sub_sequence)
print("Protein sequence:", my_protein)
print("Protein length:", len(my_protein))


# 6. REVERSE COMPLEMENT USING BIOPYTHON

the_rev_comp = my_dna.reverse_complement()

print("\nReverse complement:", the_rev_comp)


# 7. TRANSCRIPTION AND TRANSLATION OF A DNA SEQUENCE

print("\nTranscribed RNA:", my_dna.transcribe())
print("Translated protein:", my_dna.translate())


# 8. VERIFY WHETHER A SEQUENCE IS DNA, RNA, OR INVALID

def verify(sequence):
    """Identify a sequence as DNA, RNA, or Invalid."""

    sequence = sequence.strip().upper()

    if sequence and all(base in "ATCG" for base in sequence):
        return "DNA"
    elif sequence and all(base in "AUCG" for base in sequence):
        return "RNA"
    else:
        return "Invalid"


seq1 = "ATGCAGCTGTGTTACGCGAT"
seq2 = "UGGCGGAUAAGCGCA"
seq3 = "TYHGGHHHHH"

print("\nSequence validation:")
print(seq1, "is", verify(seq1))
print(seq2, "is", verify(seq2))
print(seq3, "is", verify(seq3))


# 9. REVERSE COMPLEMENT USING STRING MANIPULATION

def rev_comp_st(seq):
    """Calculate the reverse complement of DNA or RNA using strings."""

    seq = seq.strip().upper()
    sequence_type = verify(seq)

    if sequence_type == "DNA":
        complement = (
            seq.replace("A", "t")
               .replace("T", "a")
               .replace("C", "g")
               .replace("G", "c")
        )

    elif sequence_type == "RNA":
        complement = (
            seq.replace("A", "u")
               .replace("U", "a")
               .replace("C", "g")
               .replace("G", "c")
        )

    else:
        return "Invalid"

    return complement.upper()[::-1]


print("\nReverse complement using strings:")

print(
    "Reverse complement of", seq1, "is", rev_comp_st(seq1)
)
print(
    "Reverse complement of", seq2, "is", rev_comp_st(seq2)
)
print(
    "Reverse complement of", seq3, "is", rev_comp_st(seq3)
)


# 10. REVERSE COMPLEMENT USING A LIST

def rev_comp_if(seq):
    """Calculate the reverse complement using a list."""

    seq = seq.strip().upper()
    sequence_type = verify(seq)
    comp = []

    if sequence_type == "DNA":
        complement_table = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C"
        }

    elif sequence_type == "RNA":
        complement_table = {
            "A": "U",
            "U": "A",
            "C": "G",
            "G": "C"
        }

    else:
        return "Invalid"

    for base in seq:
        comp.append(complement_table[base])

    comp_rev = comp[::-1]

    return "".join(comp_rev)


print("\nReverse complement using a list:")

print(
    "Reverse complement of", seq1, "is", rev_comp_if(seq1)
)
print(
    "Reverse complement of", seq2, "is", rev_comp_if(seq2)
)
print(
    "Reverse complement of", seq3, "is", rev_comp_if(seq3)
)


# 11. TRANSLATION OF MESSENGER RNA

messenger_rna = Seq(
    "AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG"
)

print("\nMessenger RNA:", messenger_rna)
print("Translated protein:", messenger_rna.translate())