import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 100
data = {
    'CustomerID': range(1, num_customers + 1),
    'PurchaseFrequency': np.random.poisson(lam=5, size=num_customers), # Purchases per month
    'AverageOrderValue': np.random.uniform(50, 200, size=num_customers),
    'TotalSpent': np.random.normal(loc=1000, scale=500, size=num_customers)
}
df = pd.DataFrame(data)
df['TotalSpent'] = df['TotalSpent'].clip(lower=0) #Ensure no negative spending
# --- 2. Data Cleaning & Preprocessing ---
# In a real-world scenario, this would involve handling missing values, outliers, etc.
# For this synthetic data, we'll just ensure positive values for relevant columns.
# --- 3. Analysis: Customer Segmentation using Hierarchical Clustering ---
# Standardize the data for clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['PurchaseFrequency', 'AverageOrderValue', 'TotalSpent']])
# Perform hierarchical clustering
linked = linkage(df_scaled, 'ward')
# --- 4. Visualization: Dendrogram ---
plt.figure(figsize=(12, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer ID')
plt.ylabel('Distance')
plt.tight_layout()
# Save the dendrogram
output_filename = 'customer_segmentation_dendrogram.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 5.  Further Analysis (Example:  K-Means based on Dendrogram insights)---
# Based on the dendrogram, we might choose a number of clusters (e.g., 3)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)
#Analyze cluster characteristics (optional, for brevity this is omitted)
# --- 6. Visualization: Cluster characteristics (Example) ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AverageOrderValue', y='TotalSpent', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segments based on Spending')
plt.xlabel('Average Order Value')
plt.ylabel('Total Spent')
plt.tight_layout()
output_filename = 'customer_segments.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")