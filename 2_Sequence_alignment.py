# ============================================================
# SEQUENCE ALIGNMENT Methods
# ============================================================
# 1. Basic Pairwise Sequence Alignment
# 2. Display All Optimal Alignments
# 3. Global Alignment with Mismatch Penalty
# 4. Local Alignment of Chromosome and Transcript
# 5. Alignment of Transcript and Short Sequence
# 6. Mapping Two Alignments
# 7. Mapping Alignments with Undefined Sequences
# 8. Local Alignment of Similar Sequences
# 9. Display Aligner Settings and Algorithm
# 10. Calculate Alignment Scores
# 11. Alignment with Custom Scoring Parameters
# 12. Implement the Needleman-Wunsch Algorithm
# 13. Align Random DNA Sequences with Different Gap Penalties


import numpy as np
from Bio import Align
from Bio.Seq import Seq


# 1. BASIC PAIRWISE SEQUENCE ALIGNMENT

aligner = Align.PairwiseAligner()
aligner.match_score = 1.0

target = "GAACT"
query = "GAT"

print("\n1. BASIC PAIRWISE ALIGNMENT")
print("Target sequence:", target)
print("Query sequence:", query)
print("Alignment score:", aligner.score(target, query))

alignments = aligner.align(target, query)

print("\nFirst alignment:")
print(alignments[0])

print("Target:", alignments[0].target)
print("Query:", alignments[0].query)
print("Coordinates:", alignments[0].coordinates)
print("Number of aligned sequences:", len(alignments[0]))
print("Alignment length:", alignments[0].length)


# 2. DISPLAY ALL OPTIMAL ALIGNMENTS

print("\n2. ALL OPTIMAL ALIGNMENTS")

for index, alignment in enumerate(alignments, start=1):
    print(f"\nAlignment {index}:")
    print(alignment)
    print("Score:", alignment.score)


# 3. GLOBAL ALIGNMENT WITH MISMATCH PENALTY

aligner.mode = "global"
aligner.mismatch_score = -10

target = "AAACAAA"
query = "AAAGAAA"

global_alignments = aligner.align(target, query)

print("\n3. GLOBAL ALIGNMENT")
print("Target:", target)
print("Query:", query)
print("Number of optimal alignments:", len(global_alignments))

print("\nFirst global alignment:")
print(global_alignments[0])
print("Aligned regions:", global_alignments[0].aligned)

if len(global_alignments) > 1:
    print("\nSecond global alignment:")
    print(global_alignments[1])
    print("Aligned regions:", global_alignments[1].aligned)


# 4. LOCAL ALIGNMENT OF CHROMOSOME AND TRANSCRIPT

aligner.mode = "local"
aligner.open_gap_score = -1
aligner.extend_gap_score = 0

chromosome = "AAAAAAAACCCCCCCAAAAAAAAAAAGGGGGGAAAAAAAA"
transcript = "CCCCCCCGGGGGG"

local_alignments = aligner.align(chromosome, transcript)
alignment1 = local_alignments[0]

print("\n4. LOCAL ALIGNMENT")
print("Chromosome:", chromosome)
print("Transcript:", transcript)
print("Number of optimal alignments:", len(local_alignments))
print("\nBest local alignment:")
print(alignment1)
print("Alignment score:", alignment1.score)


# 5. ALIGNMENT OF TRANSCRIPT AND SHORT SEQUENCE

sequence = "CCCCGGGG"
alignments2 = aligner.align(transcript, sequence)
alignment2 = alignments2[0]

print("\n5. TRANSCRIPT AND SHORT SEQUENCE ALIGNMENT")
print("Transcript:", transcript)
print("Short sequence:", sequence)
print("Number of optimal alignments:", len(alignments2))
print("\nBest alignment:")
print(alignment2)
print("Alignment score:", alignment2.score)


# 6. MAP TWO ALIGNMENTS

mapped_alignment = alignment1.map(alignment2)

print("\n6. MAPPED ALIGNMENT")
print(mapped_alignment)
print("\nMapped alignment in PSL format:")
print(format(mapped_alignment, "psl"))


# 7. MAP ALIGNMENTS WITHOUT DISPLAYING SEQUENCE LETTERS

alignment1.target = Seq(None, len(alignment1.target))
alignment1.query = Seq(None, len(alignment1.query))

alignment2.target = Seq(None, len(alignment2.target))
alignment2.query = Seq(None, len(alignment2.query))

mapped_alignment = alignment1.map(alignment2)

print("\n7. MAPPED ALIGNMENT WITH UNDEFINED SEQUENCES")
print(format(mapped_alignment, "psl"))


# 8. LOCAL ALIGNMENT OF SIMILAR SEQUENCES

aligner.mode = "local"

target = "AGAACTC"
query = "GAACT"

score = aligner.score(target, query)
alignments = aligner.align(target, query)

print("\n8. LOCAL ALIGNMENT OF SIMILAR SEQUENCES")
print("Target:", target)
print("Query:", query)
print("Best alignment score:", score)

for index, alignment in enumerate(alignments, start=1):
    print(f"\nAlignment {index}:")
    print(alignment)


# 9. DISPLAY ALIGNER SETTINGS AND ALGORITHM

aligner = Align.PairwiseAligner(
    match_score=1.0,
    mode="local"
)

print("\n9. ALIGNER SETTINGS")
print("Algorithm:", aligner.algorithm)
print(aligner)


# 10. CALCULATE ALIGNMENT SCORES

aligner = Align.PairwiseAligner()

print("\n10. DEFAULT MATCH SCORE")
print("Default match score:", aligner.match_score)

score = aligner.score("ACGT", "ACAT")
print("Score with default settings:", score)


# 11. ALIGNMENT WITH CUSTOM SCORING PARAMETERS

aligner.match_score = 1.0
aligner.mismatch_score = -2.0
aligner.gap_score = -2.5

score = aligner.score("ACGT", "ACAT")

print("\n11. CUSTOM ALIGNMENT SCORING")
print("Match score:", aligner.match_score)
print("Mismatch score:", aligner.mismatch_score)
print("Gap score:", aligner.gap_score)
print("Alignment score:", score)


# 12. IMPLEMENT THE NEEDLEMAN-WUNSCH ALGORITHM

def nw(x, y, match=1, mismatch=1, gap=1):
    """Perform global alignment using the Needleman-Wunsch algorithm."""

    nx = len(x)
    ny = len(y)

    F = np.zeros((nx + 1, ny + 1))
    F[:, 0] = np.arange(nx + 1) * -gap
    F[0, :] = np.arange(ny + 1) * -gap

    P = np.zeros((nx + 1, ny + 1), dtype=int)

    P[1:, 0] = 3
    P[0, 1:] = 4

    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            diagonal_score = F[i - 1, j - 1] + (
                match if x[i - 1] == y[j - 1] else -mismatch
            )

            up_score = F[i - 1, j] - gap
            left_score = F[i, j - 1] - gap

            best_score = max(diagonal_score, up_score, left_score)
            F[i, j] = best_score

            if diagonal_score == best_score:
                P[i, j] |= 2
            if up_score == best_score:
                P[i, j] |= 3
            if left_score == best_score:
                P[i, j] |= 4

    i = nx
    j = ny
    aligned_x = []
    aligned_y = []

    while i > 0 or j > 0:
        direction = P[i, j]

        if i > 0 and j > 0 and direction & 2:
            aligned_x.append(x[i - 1])
            aligned_y.append(y[j - 1])
            i -= 1
            j -= 1

        elif i > 0 and direction & 1:
            aligned_x.append(x[i - 1])
            aligned_y.append("-")
            i -= 1

        elif j > 0:
            aligned_x.append("-")
            aligned_y.append(y[j - 1])
            j -= 1

        else:
            raise ValueError("Unable to trace alignment.")

    aligned_x = "".join(aligned_x[::-1])
    aligned_y = "".join(aligned_y[::-1])

    return aligned_x + "\n" + aligned_y


x = "GATTACA"
y = "GCATGCU"

print("\n12. NEEDLEMAN-WUNSCH GLOBAL ALIGNMENT")
print("Sequence 1:", x)
print("Sequence 2:", y)
print("\nAligned sequences:")
print(nw(x, y))


# 13. ALIGN RANDOM DNA SEQUENCES WITH DIFFERENT GAP PENALTIES

np.random.seed(42)

x = "".join(np.random.choice(["A", "T", "G", "C"], 50))
y = "".join(np.random.choice(["A", "T", "G", "C"], 50))

print("\n13. RANDOM DNA SEQUENCE ALIGNMENT")
print("Sequence 1:", x)
print("Sequence 2:", y)

print("\nAlignment with gap penalty 0:")
print(nw(x, y, gap=0))

print("\nAlignment with gap penalty 1:")
print(nw(x, y, gap=1))

print("\nAlignment with gap penalty 2:")
print(nw(x, y, gap=2))