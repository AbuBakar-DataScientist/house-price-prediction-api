from sklearn.datasets import fetch_california_housing
import pandas as pd

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)

df["price"] = data.target
print("Shape of the dataset:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nSummary statistics of the dataset:")
print(df.describe())