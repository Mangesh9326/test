import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("./data/23_Sharktank.csv")
# Select Numerical Columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 1. Mean, Median, Mode, Symmetry
summary_stats = df[num_cols].agg(['mean', 'median', lambda x: x.mode().iloc[0]])
summary_stats.index = ['Mean', 'Median', 'Mode']
symmetry_comments = {}
for col in num_cols:
    mean = summary_stats.loc['Mean', col]
    median = summary_stats.loc['Median', col]
    if abs(mean - median) < 1:
        symmetry_comments[col] = 'Symmetric'
    elif mean > median:
        symmetry_comments[col] = 'Right-skewed'
    else:
        symmetry_comments[col] = 'Left-skewed'
summary_df = summary_stats.T
summary_df['Symmetry'] = summary_df.index.map(symmetry_comments)
print(summary_df)

# Set Plot Style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

# 2. Bar Plot – Deals by Shark
deal_counts = df[['ashneer_deal','anupam_deal','aman_deal','namita_deal','vineeta_deal','peyush_deal','ghazal_deal']].sum()
deal_counts.plot(kind='bar', color='skyblue', title="Bar plot - Deals by Sharks")
plt.ylabel("Number of Deals")
plt.tight_layout()
plt.show()

# 3. Stacked Bar Plot – Presence vs Deals
presence = df[['ashneer_present','anupam_present','aman_present','namita_present','vineeta_present','peyush_present','ghazal_present']].sum()
stack_df = pd.DataFrame({'Present': presence, 'Deals': deal_counts})
stack_df.plot(kind='bar', stacked=True, title='Stacked Bar Plot Presence vs Deals by Sharks', colormap='viridis')
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 4. Pie Chart – Deal vs No Deal
deal_pie = df['deal'].value_counts()
plt.pie(deal_pie, labels=["Deal", "No Deal"], autopct='%1.1f%%', colors=['lightgreen','salmon'])
plt.title("Pie Chart - Overall Deal Distribution")
plt.show()

# 5. Histogram – Deal Amount
sns.histplot(df['deal_amount'], bins=20, kde=True, color='orange')
plt.title("Histogram of Deal Amount")
plt.xlabel("Deal Amount (Lakhs)")
plt.show()

# 6. Scatter Plot – Ask vs Deal Amount
sns.scatterplot(x='pitcher_ask_amount', y='deal_amount', hue='deal', data=df, palette='Set1')
plt.title("Scatter Plot - Pitcher Ask vs Deal Amount")
plt.xlabel("Ask Amount")
plt.ylabel("Deal Amount")
plt.show()

# 7. Box Plot – Equity per Shark
sns.boxplot(y='equity_per_shark', data=df, color='lightblue')
plt.title("Box Plot of Equity per Shark")
plt.ylabel("Equity (%)")
plt.show()

# 8. Line Chart – Episode vs Deal Amount
sns.lineplot(x='episode_number', y='deal_amount', data=df, marker="o")
plt.title("Line Chart - Deal Amount over Episodes")
plt.ylabel("Deal Amount")
plt.show()

# 9. Violin Plot – Deal Amount
sns.violinplot(data=df, y="deal_amount", inner="box", color="plum")
plt.title("Violin Plot of Deal Amount")
plt.ylabel("Deal Amount")
plt.show()

# 10. Swarm Plot – Ask Equity by Deal Status
sns.swarmplot(data=df, x='deal', y='ask_equity', hue='deal', palette='coolwarm', legend=False)
plt.title("Swarm Plot - Ask Equity Distribution by Deal Status")
plt.xticks([0, 1], ["No Deal", "Deal"])
plt.show()

# 11. Donut Chart – Investment Share by Sharks
total_investment = deal_counts.sum()
shark_donut = deal_counts / total_investment
plt.pie(shark_donut, labels=deal_counts.index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.4))
plt.title("Donut Chart - Investment Share by Sharks (Donut)")
plt.show()

# 12. 3D Scatter – Ask, Deal, Equity
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['pitcher_ask_amount'], df['deal_amount'], df['ask_equity'], c=df['deal'], cmap='coolwarm')
ax.set_xlabel("Ask Amount")
ax.set_ylabel("Deal Amount")
ax.set_zlabel("Ask Equity")
ax.set_title("3D Scatter: Ask vs Deal vs Equity")
plt.show()

# 13. Heatmap – Correlation
corr = df[num_cols].corr()
sns.heatmap(corr, annot=False, cmap='YlGnBu', linewidths=.5)
plt.title("Heatmap of Correlations")
plt.show()
