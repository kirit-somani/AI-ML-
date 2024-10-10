import pandas as pd

# Import data
file_path = "C:/Users/kirit/spyder files/crime index.csv"
data = pd.read_csv(file_path)

# Export data
data.to_csv('exported_data.csv', index=False)

# Display dataset details
print("Number of Rows:", data.shape[0])
print("Number of Columns:", data.shape[1])
print("First Five Rows:\n", data.head())
print("Dataset Size:", data.size)
print("Number of Missing Values:\n", data.isnull().sum())
print("Sum of Numerical Columns:\n", data.sum(numeric_only=True))
print("Average of Numerical Columns:\n", data.mean(numeric_only=True))
print("Minimum Values:\n", data.min(numeric_only=True))
print("Maximum Values:\n", data.max(numeric_only=True))
