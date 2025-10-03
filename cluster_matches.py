import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data/second_round_with_stats.csv')

# Create a match identifier
df['match_id'] = df['home_team'] + ' vs ' + df['away_team']

# Select only numerical features for clustering (excluding team names and positions)
feature_cols = [col for col in df.columns if col not in ['home_team', 'away_team', 'match_id']]
X = df[feature_cols]

# Standardize the features (important for k-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using elbow method
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.savefig('plots/elbow_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Perform k-means with optimal k (let's use k=3 as default, adjust based on elbow curve)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Add cluster centers (in original scale)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers, columns=feature_cols)
centers_df.index = [f'Cluster {i}' for i in range(optimal_k)]

# Save results
df_output = df[['match_id', 'home_team', 'away_team', 'cluster']]
#sort by cluster for better readability
df_output = df_output.sort_values(by='cluster')
df_output.to_csv('data/clustered_matches.csv', index=False)

# Print summary statistics for each cluster
print(f"K-Means Clustering Results (k={optimal_k})")
print("=" * 60)
print(f"\nTotal matches: {len(df)}")
print("\nMatches per cluster:")
print(df['cluster'].value_counts().sort_index())

print("\n" + "=" * 60)
print("Cluster Characteristics (Cluster Centers):")
print("=" * 60)
print(centers_df.round(2))

print("\n" + "=" * 60)
print("Sample matches from each cluster:")
print("=" * 60)
for i in range(optimal_k):
    cluster_matches = df[df['cluster'] == i]['match_id'].head(3).tolist()
    print(f"\nCluster {i}:")
    for match in cluster_matches:
        print(f"  - {match}")

print(f"\n\nResults saved to 'clustered_matches.csv'")
print(f"Elbow curve saved to 'elbow_curve.png'")


