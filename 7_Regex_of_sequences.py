# Program to find the regular expression of these sequences.
# open https://www.uniprot.org/ and download the 3 fasta files "Cytochrome c Homo sapiens", "Cytochrome c Mus musculus", "Cytochrome c Gallus gallus"
# create them into seq1.txt, seq2.txt and seq3.txt only take the sequence and also make an seq1.fasta, seq2.fasta and seq3.fasta with the same sequence
# Program to find the regular expression of these sequences.
def main():
    try:
        with open("seq1.txt", "r") as file1, open("seq2.txt", "r") as file2, open("seq3.txt", "r") as file3:
            s1 = file1.read()
            s2 = file2.read()
            s3 = file3.read()
        flag = 0
        len_s1 = len(s1)
        for j in range(len_s1):
            ch1 = s1[j] if j < len(s1) else ''
            ch2 = s2[j] if j < len(s2) else ''
            ch3 = s3[j] if j < len(s3) else ''
            if ch1 == ch2 and ch2 == ch3:
                flag = 1
                print(f"-{ch1}", end="")
            if ch1 != ch2 and ch2 != ch3:
                if ch1 != ch3:
                    flag = 1
                    print("-X", end="")
            if ch1 != ch2 and ch2 == ch3:
                flag = 1
                print(f"-[{ch1}{ch2}]", end="")
            if ch1 != ch2 and ch1 == ch3:
                flag = 1
                print(f"-[{ch2}{ch3}]", end="")
            if ch2 != ch3 and ch1 == ch2:
                flag = 1
                print(f"-[{ch1}{ch3}]", end="")
        if flag:
            print()
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    main() 

import os
from Bio import SeqIO
folder_path = r""
file_names = [
    "seq1.fasta",
    "seq2.fasta",
    "seq3.fasta"
]
def load_sequence(file_path):
    record = next(SeqIO.parse(file_path, "fasta"))
    return str(record.seq)
sequences = []
for filename in file_names:
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()
    sequence = load_sequence(file_path)
    sequences.append(sequence)
    print(f"Loaded {filename}, Length: {len(sequence)}")
lengths = [len(seq) for seq in sequences]
if len(set(lengths)) != 1:
    print("Warning: Sequences have different lengths.")
    print("Regex will be generated up to the shortest sequence.")
min_length = min(lengths)
sequences = [seq[:min_length] for seq in sequences]
def generate_regex(seqs):
    pattern = ""
    for chars in zip(*seqs):
        unique_chars = sorted(set(chars))
        if len(unique_chars) == 1:
            pattern += unique_chars[0]
        else:
            pattern += "[" + "".join(unique_chars) + "]"
    return pattern
regex_pattern = generate_regex(sequences)
print("\nRegular Expression Pattern:")
print(regex_pattern)