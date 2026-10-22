# DYNAMIC PROGRAMMING WITH PAIRWISE SEQUENCE ALIGNMENT
# 2. Smith-Waterman Algorithm for Local Alignment

def smith_waterman_score(seq1, seq2, match=2, mismatch=-1, gap=-2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_score = 0

    # Fill the DP table and allow local alignments to restart at zero
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                0,
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )
            max_score = max(max_score, dp[i][j])

    return max_score, dp


def print_table(dp, seq1, seq2):
    # Display the DP table with aligned columns
    width = 5
    print(" " * width + f"{'-':>{width}}", end="")
    for base in seq2:
        print(f"{base:>{width}}", end="")
    print()

    for i, row in enumerate(dp):
        label = "-" if i == 0 else seq1[i - 1]
        print(f"{label:>{width}}", end="")
        for value in row:
            print(f"{value:>{width}}", end="")
        print()


# Define sequences and scoring parameters
seq1 = "GAATTC"
seq2 = "GATAC"
match = 2
mismatch = -1
gap = -2

print("SMITH-WATERMAN LOCAL ALIGNMENT")
print("=" * 40)
print(f"Sequence 1: {seq1}")
print(f"Sequence 2: {seq2}")
print(f"Match: {match} | Mismatch: {mismatch} | Gap: {gap}")

# Calculate and display the local alignment score and DP table
score, table = smith_waterman_score(seq1, seq2, match, mismatch, gap)

print("\nDynamic Programming Table:")
print_table(table, seq1, seq2)
print(f"\nLocal Alignment Score: {score}")
