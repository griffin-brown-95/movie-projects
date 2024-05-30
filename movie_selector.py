import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the data
df_url = 'https://storage.googleapis.com/data-projects/top_1000_popular_movies_tmdb.csv'
df = pd.read_csv(df_url, lineterminator='\n')
df = df.drop(df.columns[0], axis=1)

# Compute the TF-IDF matrix and cosine similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Normalize the vote_average, vote_count, and popularity columns
scaler = MinMaxScaler()
df[['vote_average', 'vote_count', 'popularity']] = scaler.fit_transform(df[['vote_average', 'vote_count', 'popularity']])

# Calculate time decay
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
current_date = pd.to_datetime('today')
df['days_since_release'] = (current_date - df['release_date']).dt.days
df['time_decay'] = np.exp(-df['days_since_release'] / 365)

def get_weighted_similarity(idx, cosine_sim=cosine_sim, df=df, weight_votes=0.02, weight_popularity=0.02):
    sim_scores = list(enumerate(cosine_sim[idx]))
    vote_avg_scores = df['vote_average'].values
    vote_count_scores = df['vote_count'].values
    popularity_scores = df['popularity'].values
    
    weighted_scores = []
    for i, score in sim_scores:
        weighted_score = score + \
                        weight_votes * vote_avg_scores[i] + \
                        weight_votes * vote_count_scores[i] + \
                        weight_popularity * popularity_scores[i]
        weighted_scores.append((i, weighted_score))
    
    return weighted_scores

def get_recommendations_table(title, cosine_sim=cosine_sim, df=df, weight_votes=0.05, weight_popularity=0.05):
    try:
        idx = df[df['title'].str.contains(title, case=False, na=False)].index[0]
    except IndexError:
        return "<p>No recommendations found for the given title.</p>"
    
    print(f"Index of the movie '{title}': {idx}")
    
    sim_scores = get_weighted_similarity(idx, cosine_sim, df, weight_votes, weight_popularity)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    #print(f"Movie indices for recommendations: {movie_indices}")
    #print(f"Similarity scores: {sim_scores}")
    
    recommendations = df['title'].iloc[movie_indices]
    recommendations_df = recommendations.to_frame(name='Title')
    html_table = recommendations_df.to_html(classes='table table-striped', index=False)
    html_table = html_table.replace('<th>Title</th>', '<th style="text-align: left;">Title</th>')
    
    return html_table



#html_table = get_recommendations_table("Fast X")
#print(html_table)