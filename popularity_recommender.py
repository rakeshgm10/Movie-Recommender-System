import pandas as pd

# Load datasets
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Compute average rating and rating count
movie_ratings = ratings.groupby("movieId")["rating"].mean()
movie_counts = ratings.groupby("movieId")["rating"].count()

# Merge with movie details
movie_stats = pd.DataFrame({"avg_rating": movie_ratings, "rating_count": movie_counts}).reset_index()
popular_movies = movie_stats.merge(movies, on="movieId")

# Filter movies with at least 50 ratings and sort by average rating
popular_movies = popular_movies[popular_movies["rating_count"] > 50].sort_values(by="avg_rating", ascending=False)

# Display top 10 movies
print(popular_movies.head(10))
