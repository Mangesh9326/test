import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

music_df = pd.read_csv('data/7_music_data.csv')

# Create user-track matrix using listen_count as implicit rating
user_track_matrix = music_df.pivot_table(index='user_id', columns='track_name', values='listen_count',
fill_value=0)

def user_user_recommendation_music(user_id):
    if user_id not in user_track_matrix.index:
        return "User not found in data."

    similarity = cosine_similarity(user_track_matrix)
    sim_df = pd.DataFrame(similarity, index=user_track_matrix.index, columns=user_track_matrix.index)
    
    # Get top 5 similar users
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:6].index
    recommendations = []
    
    for sim_user in similar_users:
        listened = user_track_matrix.loc[sim_user]
        # Get tracks user listened more than 1 time
        recs = listened[listened > 1].index.tolist()
        recommendations.extend(recs)
        
    # Filter out tracks user already listened
    listened_by_user = user_track_matrix.loc[user_id][user_track_matrix.loc[user_id] > 0].index
    final_recs = [track for track in recommendations if track not in listened_by_user]
    return list(set(final_recs))[:10]

def content_based_recommendation_music(track_title):
    music_df['track_genre'] = music_df['track_genre'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(music_df['track_genre'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(music_df.index, index=music_df['track_name']).drop_duplicates()
    if track_title not in indices:
    # Try to find close match
        matches = [t for t in indices.index if track_title.lower() in t.lower()]
        if not matches:
            return "Track title not found."
            track_title = matches[0]
            idx = indices[track_title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            track_indices = [i[0] for i in sim_scores[1:11]]
            return f"\nTop 10 similar tracks to '{track_title}':\n" + "\n".join(music_df['track_name'].iloc[track_indices])

def popularity_based_recommendation_music():
    popular_tracks = music_df.groupby('track_name').agg({'popularity': 'mean', 'listen_count': 'sum'})
    popular_tracks = popular_tracks.sort_values(by=['popularity', 'listen_count'], ascending=False)
    popular = popular_tracks.head(10).index
    return "Top 10 Popular Tracks:\n\n" + "\n".join(popular)

def main():
    print("Available user IDs in dataset:", music_df['user_id'].unique())
    print("Available tracks in dataset:", music_df['track_name'].unique())
    print("\nChoose Recommendation Method:")
    print("1. User-User Collaborative Filtering")
    print("2. Content-Based Filtering (Genre-based)")
    print("3. Popularity-Based Recommendation")
    
    choice = input("Enter choice (1/2/3): ")
    if choice == '1':
        user_id = int(input("Enter user ID (1-20): "))
        recommendations = user_user_recommendation_music(user_id)
        print(f"\nRecommended tracks for User {user_id}:", recommendations)
    elif choice == '2':
        title = input("Enter track title (or part of it): ")
        print(content_based_recommendation_music(title))
    elif choice == '3':
        print(popularity_based_recommendation_music())
    else:
        print("Invalid choice. Please select 1, 2, or 3.")
if __name__ == "__main__":
    main()
