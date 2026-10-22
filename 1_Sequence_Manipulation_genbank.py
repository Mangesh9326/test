# BIOINFORMATICS: SEQUENCE MANIPULATION USING BIOPYTHON
# RETRIEVAL OF NUCLEOTIDE SEQUENCE FROM GENBANK
# Retrieval of Protein sequence from GenBank
# Nucleotide:
# https://www.ncbi.nlm.nih.gov/nuccore/NM_000546
# Protein:
# https://www.ncbi.nlm.nih.gov/protein/NP_000537
# Read and parse sequence data from files
# Perform basic sequence manipulations (e.g., reverse complement, translation)


import random
from Bio.Seq import Seq
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

Entrez.email = ""


# DNA sequence manipulation
dna = Seq(
    "AATTCTAGGCCGTCCTGGCGATAAAGGTGAGCCAGGTAAACATGTGCCCGGAGTTGCACCACCAGGACCA"
    "CCAGGTCGACCGGGTGATATGGGACCACCTGGACCACCTGGGCCACCAGGCGAGAAAGGAAAACCTGGAA"
    "CAAAAGGCCCAATAGGACCACCGGGAGACCAAGGAAATCCAGGTATTTATATTCGTAGTTAATGCTTGAT"
    "TCGAGAATT"
)

print("\nDNA sequence:", dna)
print("Complement:", dna.complement())
print("Reverse complement:", dna.reverse_complement())


# Transcribe DNA into RNA
rna = dna.transcribe()
print("\nRNA:", rna)


# Translate DNA and RNA into protein
print("DNA to protein:", dna.translate())
print("RNA to protein:", rna.translate())


# Generate a random DNA sequence
def generate_random_sequence(sequence_length, probability_distribution):
    sequence = ""
    for i in range(sequence_length):
        sequence += random.choices(
            ["A", "C", "G", "T"],
            weights=probability_distribution,
            k=1
        )[0]
    return sequence


our_random_seq = generate_random_sequence(100, [0.25, 0.25, 0.25, 0.25])
print("\nRandom DNA sequence:", our_random_seq)


# Perform sequence slicing and calculate length
my_dna = Seq("ATCGATGATAGGATA")
my_protein = Seq("KGLYQRGD")

a_sub_sequence = my_dna[3:7]
the_sequence_length = len(my_dna)

print("\nDNA sequence:", my_dna)
print("DNA length:", the_sequence_length)
print("Subsequence:", a_sub_sequence)
print("Protein sequence:", my_protein)
print("Protein length:", len(my_protein))


# Perform reverse complement using Biopython
the_rev_comp = my_dna.reverse_complement()
print("\nReverse complement:", the_rev_comp)


# Transcribe and translate another DNA sequence
print("Transcribed RNA:", my_dna.transcribe())
print("Translated protein:", my_dna.translate())


# Verify whether a sequence is DNA, RNA, or invalid
def verify(sequence):
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


# Calculate reverse complement using string manipulation
def rev_comp_st(seq):
    seq = seq.strip().upper()
    sequence_type = verify(seq)

    if sequence_type == "DNA":
        complement = seq.translate(str.maketrans("ATCG", "TAGC"))
    elif sequence_type == "RNA":
        complement = seq.translate(str.maketrans("AUCG", "UAGC"))
    else:
        return "Invalid"

    return complement[::-1]


print("\nReverse complement using strings:")
print(seq1, ":", rev_comp_st(seq1))
print(seq2, ":", rev_comp_st(seq2))
print(seq3, ":", rev_comp_st(seq3))


# Calculate reverse complement using a list
def rev_comp_if(seq):
    seq = seq.strip().upper()
    sequence_type = verify(seq)

    if sequence_type == "DNA":
        complement_table = {"A": "T", "T": "A", "C": "G", "G": "C"}
    elif sequence_type == "RNA":
        complement_table = {"A": "U", "U": "A", "C": "G", "G": "C"}
    else:
        return "Invalid"

    comp = [complement_table[base] for base in seq]
    return "".join(comp[::-1])


print("\nReverse complement using a list:")
print(seq1, ":", rev_comp_if(seq1))
print(seq2, ":", rev_comp_if(seq2))
print(seq3, ":", rev_comp_if(seq3))


# Translate messenger RNA into protein
messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")
print("\nMessenger RNA:", messenger_rna)
print("Translated protein:", messenger_rna.translate())


# Retrieve nucleotide sequence from GenBank
nucleotide_id = "NM_000546"

try:
    with Entrez.efetch(
        db="nuccore",
        id=nucleotide_id,
        rettype="gb",
        retmode="text"
    ) as handle:
        nucleotide_record = SeqIO.read(handle, "genbank")

    print("\nNucleotide accession:", nucleotide_record.id)
    print("Description:", nucleotide_record.description)
    print("Length:", len(nucleotide_record.seq))
    print("Nucleotide sequence:", nucleotide_record.seq)

except Exception as error:
    print("Nucleotide retrieval error:", error)


# Retrieve protein sequence from GenBank
protein_id = "NP_000537"
protein_record = None

try:
    with Entrez.efetch(
        db="protein",
        id=protein_id,
        rettype="gb",
        retmode="text"
    ) as handle:
        protein_record = SeqIO.read(handle, "genbank")

    print("\nProtein accession:", protein_record.id)
    print("Description:", protein_record.description)
    print("Length:", len(protein_record.seq))
    print("Protein sequence:", protein_record.seq)

except Exception as error:
    print("Protein retrieval error:", error)


# Read and parse sequences from a FASTA file
fasta_file = "sequences.fasta"

try:
    for record in SeqIO.parse(fasta_file, "fasta"):
        print("\nFASTA ID:", record.id)
        print("Description:", record.description)
        print("Length:", len(record.seq))
        print("Sequence:", record.seq)

except FileNotFoundError:
    print("\nFASTA file not found:", fasta_file)


# Read and parse sequences from a GenBank file
genbank_file = "sequences.gb"

try:
    for record in SeqIO.parse(genbank_file, "genbank"):
        print("\nGenBank ID:", record.id)
        print("Description:", record.description)
        print("Organism:", record.annotations.get("organism", "Unknown"))
        print("Length:", len(record.seq))
        print("Sequence:", record.seq)

except FileNotFoundError:
    print("\nGenBank file not found:", genbank_file)
