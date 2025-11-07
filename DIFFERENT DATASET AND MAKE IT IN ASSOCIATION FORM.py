#WORKING ON DIFFERENT DATASET TO IT LOOK LIKE ASSOCIATION FOR GENERATING RULES
#IPL ASSOCIATION RULES

# IPL ASSOCIATION RULES (SUPER SHORT VERSION)
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("data/ipl_winners_dataset_remix_v2.csv")
trans = [[str(r.Season), r.QuarterFinal, r.Semifinal, r.Final] for _, r in df.iterrows()]
trans = [list(dict.fromkeys([str(x) for x in t if pd.notna(x) and str(x).strip()])) for t in trans]

print("Cleaned Transactions:")
[print(t) for t in trans]

te = TransactionEncoder()
df_ohe = pd.DataFrame(te.fit(trans).transform(trans), columns=te.columns_)
rules = association_rules(apriori(df_ohe, 0.1, True), metric="confidence", min_threshold=0.3)

if rules.empty: print("\nNo rules found.")
else:
    print("\nAssociation Rules:")
    for i, r in rules.reset_index(drop=True).iterrows():
        A, B = list(r.antecedents), list(r.consequents)
        num, den = int(df_ohe[A + B].all(1).sum()), int(df_ohe[A].all(1).sum())
        print(f"Rule {i+1}: {', '.join(A)} → {', '.join(B)} = {num}/{den} = {num/den:.2f}")
