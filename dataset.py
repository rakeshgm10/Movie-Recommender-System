import urllib.request
import zipfile
import os

# URL of the dataset
url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
dataset_path = "ml-latest-small.zip"

# Download the dataset
print("Downloading dataset...")
urllib.request.urlretrieve(url, dataset_path)

# Extract the dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(".")

# Remove ZIP file after extraction
os.remove(dataset_path)

print("Dataset downloaded and extracted successfully!")
