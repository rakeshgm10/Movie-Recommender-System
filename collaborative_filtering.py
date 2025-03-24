from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse
import pandas as pd

# Load datasets
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Define rating scale (0.5 - 5)
reader = Reader(rating_scale=(0.5, 5))

# Load dataset for Surprise
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

# Evaluate RMSE (lower is better)
predictions = model.test(testset)
print("RMSE:", rmse(predictions))
