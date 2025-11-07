import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- USER INPUT ---
try:
    min_support = float(input("Enter minimum support (e.g., 0.01): "))
except ValueError:
    min_support = 0.01
    print("Invalid input. Defaulting to 0.01")

try:
    min_confidence = float(input("Enter minimum confidence (e.g., 0.2): "))
except ValueError:
    min_confidence = 0.2
    print("Invalid input. Defaulting to 0.2")

data = pd.read_csv("data/8_groceries.csv")
print(data.head())
print(data.columns)

transactions = data.values.astype(str).tolist()
transactions = [[item for item in row if item != 'nan'] for row in transactions]
print("Sample transactions:")
print(transactions[:10])

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head(5))
print("Shape of one-hot encoded dataframe:", df.shape)

frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
print(f"Number of frequent itemsets found: {frequent_itemsets.shape[0]}")

plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
colors = sns.color_palette("pastel", 15)
sns.barplot(
    x='itemsets',
    y='support',
    hue='itemsets',
    data=frequent_itemsets.nlargest(n=15, columns='support'),
    palette=colors
)
plt.title(f'Top 15 Frequent Itemsets (min_support={min_support})')
plt.show()

# --- ASSOCIATION RULES ---
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

if rules.empty:
    print(f"No rules found with support >= {min_support} and confidence >= {min_confidence}")
else:
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
    rules["rule"] = rules.apply(lambda row: f"{set(row['antecedents'])} -> {set(row['consequents'])}", axis=1)
    rules["antecedent"] = rules["antecedents"].apply(lambda x: set(x))

    print("\nSample rules with 'rule' and 'antecedent' columns:")
    print(rules[['antecedent', 'rule', 'support', 'confidence', 'lift']].head(10))

    print("\n\n---------------------ACCURACY METRICS-------------------------------")

    # Show rules sorted by different metrics
    metrics_thresholds = {
        "lift": 1,
        "confidence": min_confidence,
        "leverage": 0,
        "conviction": 0
    }

    for metric, threshold in metrics_thresholds.items():
        print(f"\nRules sorted by {metric}:")
        metric_rules = association_rules(frequent_itemsets, metric=metric, min_threshold=threshold)

        if metric_rules.empty:
            print(f"No rules found for metric '{metric}' with min_threshold={threshold}")
            continue

        metric_rules["rule"] = metric_rules.apply(
            lambda row: f"{set(row['antecedents'])} -> {set(row['consequents'])}", axis=1
        )
        metric_rules["antecedent"] = metric_rules["antecedents"].apply(lambda x: set(x))

        if metric == "confidence":
            print(metric_rules[['antecedent', 'rule', 'support', 'confidence']]
                  .sort_values(by='confidence', ascending=False)
                  .head())
        else:
            print(metric_rules[['antecedent', 'rule', 'support', 'confidence', metric]]
                  .sort_values(by=metric, ascending=False)
                  .head())
