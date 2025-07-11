import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data ---
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# --- Content-Based Setup ---
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

# --- Collaborative Filtering Setup ---
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

# --- UI Design ---
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🍿 Movie Recommender System</h1>
    <p style='text-align: center; color: gray;'>Get personalized movie suggestions based on your preferences!</p>
""", unsafe_allow_html=True)

# --- Tabs for Type of Recommender ---
tab1, tab2 = st.tabs(["🎯 Content-Based", "🤝 Collaborative Filtering"])

with tab1:
    st.subheader("🎬 Content-Based Recommender")
    movie_list = movies['title'].sort_values().tolist()
    selected_movie = st.selectbox("Pick a movie you like:", movie_list)

    if st.button("🔍 Recommend based on this movie"):
        recs = content_recommend(selected_movie)
        st.success(f"Top 5 movies similar to **{selected_movie}**:")
        for i, movie in enumerate(recs, 1):
            st.write(f"**{i}. {movie}**")

with tab2:
    st.subheader("👤 Collaborative Filtering Recommender")
    user_ids = sorted(ratings['userId'].unique())
    selected_user = st.selectbox("Select a user ID (based on ratings data):", user_ids)

    if st.button("🎯 Recommend for this user"):
        recs = cf_recommend(int(selected_user))
        st.success(f"Top 5 movies for **User ID {selected_user}**:")
        for i, movie in enumerate(recs, 1):
            st.write(f"**{i}. {movie}**")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 12px;'>Made By Vishruth</p>", unsafe_allow_html=True)
