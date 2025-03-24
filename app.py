import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies dataset
movies = pd.read_csv("ml-latest-small/movies.csv")
movies["genres"] = movies["genres"].fillna("")

# Convert genres into numerical format
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies
def recommend_content(movie_title, movies, cosine_sim):
    idx = movies[movies["title"] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["title", "genres"]]

# Streamlit UI
st.title("Movie Recommender System")
movie_name = st.text_input("Enter a movie name")

if st.button("Recommend"):
    recommendations = recommend_content(movie_name, movies, cosine_sim)
    st.write(recommendations)
