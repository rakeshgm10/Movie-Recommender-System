import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("ml-latest-small/movies.csv")

# TF-IDF Vectorization on genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_movies(movie_name):
    if movie_name not in movies["title"].values:
        return ["Movie not found in dataset. Try another."]

    idx = movies.index[movies["title"] == movie_name][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "genres"]]

# Streamlit UI
st.title("🎬 Movie Recommender System")

# Input field
movie_name = st.text_input("Enter a movie name", "Titanic (1997)")

# Add an "Apply" button
if st.button("Apply"):
    recommendations = recommend_movies(movie_name)
    st.write("### ✅ These are the recommended movies:")
    st.dataframe(recommendations)
