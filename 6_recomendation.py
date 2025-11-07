import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast

df = pd.read_csv("data/6_games.csv")
df = df.drop_duplicates(subset='Title').reset_index(drop=True)

for col in ['Genres', 'Team', 'Summary']:
    df[col] = df[col].fillna('')

# Convert list-like strings to plain text
df['Genres'] = df['Genres'].apply(lambda x: ' '.join(ast.literal_eval(x)) if x.startswith('[') else x)
df['Team'] = df['Team'].apply(lambda x: ' '.join(ast.literal_eval(x)) if x.startswith('[') else x)

# Combine text columns into a single feature
df['combined'] = df['Genres'] + ' ' + df['Team'] + ' ' + df['Summary']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return f"Game '{title}' not found in dataset."

    # Handle duplicates safely
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:10]

    game_indices = [i for i, _ in sim_scores]
    return df.iloc[game_indices][['Title', 'Genres', 'Team', 'Rating']]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(get_recommendations("Need for Speed: Hot Pursuit"))
