import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

movies['genres'] = movies['genres'].str.replace('|', ' ')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def content_recommend(title, n=5):
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:n+1]]
    return movies['title'].iloc[movie_indices].tolist()


user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def cf_recommend(user_id, n=5):
    if user_id not in user_similarity_df.index:
        return ["User not found."]
    sim_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(10).index
    sim_ratings = user_movie_matrix.loc[sim_users]
    avg_ratings = sim_ratings.mean().sort_values(ascending=False)
    watched = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    recommendations = avg_ratings.drop(watched).head(n).index
    return movies[movies['movieId'].isin(recommendations)]['title'].tolist()

# --- Streamlit UI ---
st.title(" Unified Movie Recommender System By Vishruth")

option = st.radio("Choose recommendation type:", ["Content-Based", "Collaborative Filtering"])

if option == "Content-Based":
    movie_list = movies['title'].sort_values().tolist()
    selected_movie = st.selectbox("Select a movie:", movie_list)

    if st.button("Recommend"):
        recs = content_recommend(selected_movie)
        st.subheader("Top 5 Recommendations:")
        for i, movie in enumerate(recs, 1):
            st.write(f"{i}. {movie}")

elif option == "Collaborative Filtering":
    user_ids = sorted(ratings['userId'].unique())
    selected_user = st.selectbox("Select a user ID:", user_ids)

    if st.button("Recommend"):
        recs = cf_recommend(int(selected_user))
        st.subheader("Top 5 Recommendations:")
        for i, movie in enumerate(recs, 1):
            st.write(f"{i}. {movie}")
