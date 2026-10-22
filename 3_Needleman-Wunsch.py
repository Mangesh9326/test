# DYNAMIC PROGRAMMING WITH PAIRWISE SEQUENCE ALIGNMENT
# 1. Needleman-Wunsch Algorithm for Global Alignment

def needleman_wunsch_score(seq1, seq2, match=2, mismatch=-1, gap=-2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column with gap penalties
    for i in range(1, m + 1):
        dp[i][0] = i * gap
    for j in range(1, n + 1):
        dp[0][j] = j * gap

    # Fill the DP table using match, mismatch, and gap scores
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + score,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )

    return dp[m][n], dp


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

print("NEEDLEMAN-WUNSCH GLOBAL ALIGNMENT")
print("=" * 40)
print(f"Sequence 1: {seq1}")
print(f"Sequence 2: {seq2}")
print(f"Match: {match} | Mismatch: {mismatch} | Gap: {gap}")

# Calculate and display the global alignment score and DP table
score, table = needleman_wunsch_score(seq1, seq2, match, mismatch, gap)

print("\nDynamic Programming Table:")
print_table(table, seq1, seq2)
print(f"\nGlobal Alignment Score: {score}")