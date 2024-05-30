import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the data
df_url = 'https://storage.googleapis.com/data-projects/top_1000_popular_movies_tmdb.csv'
df = pd.read_csv(df_url, lineterminator='\n')
df.to_csv('example.csv')
df = df.drop(df.columns[0], axis=1)

tfidf = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

scaler = MinMaxScaler()
df[['vote_average', 'vote_count', 'popularity']] = scaler.fit_transform(df[['vote_average', 'vote_count', 'popularity']])

def get_weighted_similarity(idx, 
                            cosine_sim=cosine_sim, 
                            df=df, 
                            weight_votes=0.02, 
                            weight_popularity=0.02):
    sim_scores = list(enumerate(cosine_sim[idx]))
    vote_avg_scores = df['vote_average'].values
    vote_count_scores = df['vote_count'].values
    popularity_scores = df['popularity'].values
    
    # Calculate the weighted score
    weighted_scores = []
    for i, score in sim_scores:
        weighted_score = score + \
                        weight_votes * vote_avg_scores[i] + \
                        weight_votes * vote_count_scores[i] + \
                        weight_popularity * popularity_scores[i]
        weighted_scores.append((i, weighted_score))
    
    return weighted_scores

def get_recommendations(title, 
                        cosine_sim=cosine_sim, 
                        df=df, 
                        weight_votes=0.05, 
                        weight_popularity=0.05):
    idx = df[df['title'] == title].index[0]
    # adding in similarity by scores
    sim_scores = get_weighted_similarity(idx, 
                                         cosine_sim, 
                                         df, 
                                         weight_votes, 
                                         weight_popularity,)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['title']
    recommendations = get_recommendations(movie_title)
    return render_template('result.html', movie_title=movie_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)