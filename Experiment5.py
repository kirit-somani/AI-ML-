import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\kirit\spyder files\crime index.csv"
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Number of Rows:", data.shape[0])
print("Number of Columns:", data.shape[1])
print("\nFirst Five Rows:\n", data.head())
print("\nDataset Size:", data.size)
print("\nNumber of Missing Values:\n", data.isnull().sum())
print("\nSummary Statistics:\n", data.describe())

# Visualizing missing data
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Visualize distributions of numerical columns
data.hist(bins=30, figsize=(15, 10))
plt.suptitle('Distributions of Numerical Columns')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
