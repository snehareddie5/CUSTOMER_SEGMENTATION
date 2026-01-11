# STEP 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# STEP 2: Load the dataset
data = pd.read_csv("Mall_Customers.csv")

# STEP 3: Display first few rows
print("First 5 rows of the dataset:")
print(data.head())

# STEP 4: Display dataset information
print("\nDataset information:")
print(data.info())

# STEP 5: Select features for clustering
# Using correct column names from your dataset
X = data[['Annual Income', 'Spending Score']]

# STEP 6: Find optimal number of clusters using Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.show()

# STEP 7: Apply K-Means with optimal clusters (usually 5)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# STEP 8: Add cluster labels to the dataset
data['Cluster'] = y_kmeans

print("\nDataset with Cluster labels:")
print(data.head())

# STEP 9: Visualize the clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()

# STEP 10: Save the segmented dataset
data.to_csv("segmented_customers.csv", index=False)

print("\nSegmented customer data saved as 'segmented_customers.csv'")
