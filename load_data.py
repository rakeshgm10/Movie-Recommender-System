import pandas as pd

# Load datasets
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Display first few rows
print(movies.head())
print(ratings.head())
