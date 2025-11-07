#FP Tree : based on reg dataset


import csv
import pyfpgrowth

# === Step 1: Read CSV ===
with open('data/movies.csv', encoding="utf8", newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

# === Step 2: Clean and prepare data ===
data.pop(0)  # remove header row

transactions = []
for row in data:
    if len(row) >= 3 and row[2].strip():
        genres = row[2].split('|')
        transactions.append(genres)

print("Some transactions:")
for t in transactions[:10]:
    print(t)

# === Step 3: Find frequent patterns ===
min_support = 2
patterns = pyfpgrowth.find_frequent_patterns(transactions, min_support)
print("\nFrequent Patterns (top 10):")
for p, s in list(patterns.items())[:10]:
    print(f"{p}: {s}")

# === Step 4: Generate association rules ===
min_confidence = 0.3  # lower confidence threshold
rules = pyfpgrowth.generate_association_rules(patterns, min_confidence)

print("\nRules output:")
if not rules:
    print("No rules found — try lowering confidence or support.")
else:
    for base, (add, conf) in list(rules.items())[:10]:
        print(f"{base} -> {add}  [confidence: {conf:.2f}]")

