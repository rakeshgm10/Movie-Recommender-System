import pandas as pd
from surprise import Dataset, Reader, SVD

# Load datasets
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Define rating scale
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train SVD model
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# Recommend movies for a given user
def recommend_movies(user_id, model, movies, ratings):
    unseen_movies = movies[~movies["movieId"].isin(ratings[ratings["userId"] == user_id]["movieId"])]
    unseen_movies["predicted_rating"] = unseen_movies["movieId"].apply(lambda x: model.predict(user_id, x).est)
    return unseen_movies.sort_values("predicted_rating", ascending=False).head(10)

# Get recommendations for user 1
recommendations = recommend_movies(1, model, movies, ratings)
print(recommendations)
