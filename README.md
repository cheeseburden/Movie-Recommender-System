# 🎬 Movie Recommender System

An interactive movie recommendation app built using **Python**, **Streamlit**, and **pandas**, all from a **single Jupyter notebook**.

It supports:
- ✅ **Content-Based Filtering** (based on movie genres)
- ✅ **Collaborative Filtering** (based on user ratings)

---

## 🎯 Features

- 🔍 **Content-Based Recommender** using TF-IDF and cosine similarity on genres
- 🤝 **Collaborative Filtering Recommender** using user-user cosine similarity
- 🧠 Modern UI with Streamlit and emoji-enhanced navigation
- 📎 Built from one notebook, no manual Python modules required
- 💬 Ready for expansion: TMDB API, hybrid models, or user login

---

## 🧠 How It Works

### 📌 Content-Based Filtering:
- Preprocess genres using TF-IDF vectorizer
- Compute cosine similarity between all movies
- Recommend movies with highest similarity to the selected movie

### 📌 Collaborative Filtering:
- Build a **user-movie rating matrix** from MovieLens data
- Compute **cosine similarity between users**
- Recommend top-rated unseen movies from similar users

### 🧪 Streamlit UI:
- Renders two tabs:
  - 🎯 Content-Based
  - 🤝 Collaborative Filtering
- Cleanly separates logic and lets the user pick recommendation mode

### 🚀 Run the App
- Open and run Movie_Recommender.ipynb
- It will generate app.py using:%%writefile app.py

- streamlit run app.py

### 📁 Dataset Used
- We use the official MovieLens Latest Small Dataset, specifically:
  - movies.csv — movie titles and genres
  - ratings.csv — user-movie ratings

---
## 🧑‍💻 Author
Vishruth
