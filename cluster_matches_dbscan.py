import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# Read the CSV file
df = pd.read_csv('data/second_round_with_stats.csv')

# Create a match identifier
df['match_id'] = df['home_team'] + ' vs ' + df['away_team']

# Select only numerical features for clustering (excluding team names and positions)
feature_cols = [col for col in df.columns if col not in ['home_team', 'away_team', 'match_id']]
X = df[feature_cols]

print(f"Original feature space: {X.shape[1]} dimensions")
print(f"Number of samples: {X.shape[0]}")
print()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality (helps DBSCAN work better)
pca = PCA(n_components=0.80)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

print(f"After PCA: {X_pca.shape[1]} dimensions (explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance)")
print()

# Use k-distance graph to help determine optimal eps
k_values = [3, 5, 7]

fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5))

for idx, k in enumerate(k_values):
    neighbors = NearestNeighbors(n_neighbors=k+1)
    neighbors_fit = neighbors.fit(X_pca)
    distances, indices = neighbors_fit.kneighbors(X_pca)
    
    # Sort distances (exclude first column which is distance to self)
    distances = np.sort(distances[:, k], axis=0)
    
    # Plot k-distance graph
    axes[idx].plot(distances)
    axes[idx].set_xlabel('Points sorted by distance')
    axes[idx].set_ylabel(f'{k}-th Nearest Neighbor Distance')
    axes[idx].set_title(f'K-distance Graph (k={k})')
    axes[idx].grid(True)
    axes[idx].axhline(y=np.percentile(distances, 90), color='r', linestyle='--', label='90th percentile')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('plots/k_distance_graph.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate suggested eps range from k-distance
neighbors = NearestNeighbors(n_neighbors=6)
neighbors_fit = neighbors.fit(X_pca)
distances, _ = neighbors_fit.kneighbors(X_pca)
distances = np.sort(distances[:, 5], axis=0)

# Use percentiles to suggest eps range
eps_min = np.percentile(distances, 50)
eps_max = np.percentile(distances, 95)
suggested_eps_range = np.linspace(eps_min, eps_max, 15)

print(f"Suggested eps range: {eps_min:.3f} to {eps_max:.3f}")
print()

# Test different epsilon values with suggested range
eps_values = suggested_eps_range
min_samples_values = [3, 5, 7, 10]

results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels)
        
        # Calculate silhouette score only if we have at least 2 clusters and not all points are noise
        if n_clusters >= 2 and n_noise < len(labels) * 0.9:  # Allow up to 90% noise
            try:
                score = silhouette_score(X_pca, labels)
            except:
                score = -1
        else:
            score = -1
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette': score
        })

results_df = pd.DataFrame(results)

# Print all configurations that found clusters
print("All parameter combinations that found clusters:")
print("=" * 100)
found_clusters = results_df[results_df['n_clusters'] > 0].sort_values('silhouette', ascending=False)
if not found_clusters.empty:
    print(found_clusters.to_string(index=False))
else:
    print("No configurations found any clusters!")
print()

# Print top configurations by silhouette score
print("Top 10 parameter combinations by silhouette score:")
print("=" * 100)
top_results = results_df[results_df['silhouette'] > 0].nlargest(10, 'silhouette')
if not top_results.empty:
    print(top_results.to_string(index=False))
else:
    print("No configurations with valid silhouette scores found.")
print()

# Plot parameter exploration
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Number of clusters vs epsilon
for min_samples in min_samples_values:
    data = results_df[results_df['min_samples'] == min_samples]
    axes[0, 0].plot(data['eps'], data['n_clusters'], marker='o', label=f'min_samples={min_samples}')
axes[0, 0].set_xlabel('Epsilon (eps)')
axes[0, 0].set_ylabel('Number of Clusters')
axes[0, 0].set_title('DBSCAN: Number of Clusters vs Epsilon')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Silhouette score vs epsilon
for min_samples in min_samples_values:
    data = results_df[results_df['min_samples'] == min_samples]
    axes[0, 1].plot(data['eps'], data['silhouette'], marker='o', label=f'min_samples={min_samples}')
axes[0, 1].set_xlabel('Epsilon (eps)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('DBSCAN: Silhouette Score vs Epsilon')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Noise ratio vs epsilon
for min_samples in min_samples_values:
    data = results_df[results_df['min_samples'] == min_samples]
    axes[1, 0].plot(data['eps'], data['noise_ratio'], marker='o', label=f'min_samples={min_samples}')
axes[1, 0].set_xlabel('Epsilon (eps)')
axes[1, 0].set_ylabel('Noise Ratio')
axes[1, 0].set_title('DBSCAN: Noise Ratio vs Epsilon')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: Scatter of clusters vs noise ratio colored by silhouette
scatter = axes[1, 1].scatter(results_df['n_clusters'], results_df['noise_ratio'], 
                            c=results_df['silhouette'], cmap='RdYlGn', s=100, alpha=0.6)
axes[1, 1].set_xlabel('Number of Clusters')
axes[1, 1].set_ylabel('Noise Ratio')
axes[1, 1].set_title('Clusters vs Noise Ratio (colored by Silhouette Score)')
axes[1, 1].grid(True)
plt.colorbar(scatter, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('plots/dbscan_parameter_exploration.png', dpi=300, bbox_inches='tight')
plt.close()

# Choose optimal parameters based on best balance
# Prioritize: silhouette score > 0.2, noise ratio < 0.3, and reasonable number of clusters
good_configs = results_df[
    (results_df['silhouette'] > 0.1) & 
    (results_df['n_clusters'] >= 2) & 
    (results_df['noise_ratio'] < 0.4)
]

if not good_configs.empty:
    best_config = good_configs.nlargest(1, 'silhouette')
    optimal_eps = best_config['eps'].values[0]
    optimal_min_samples = int(best_config['min_samples'].values[0])
else:
    # Try to find any configuration with clusters
    has_clusters = results_df[results_df['n_clusters'] > 0]
    if not has_clusters.empty:
        # Pick the one with best balance
        has_clusters['score'] = has_clusters['n_clusters'] * (1 - has_clusters['noise_ratio'])
        best_config = has_clusters.nlargest(1, 'score')
        optimal_eps = best_config['eps'].values[0]
        optimal_min_samples = int(best_config['min_samples'].values[0])
    else:
        # Last resort: use middle of range with small min_samples
        optimal_eps = np.median(eps_values)
        optimal_min_samples = 3
        print("WARNING: Could not find good parameters, using fallback values")

print(f"Selected optimal parameters: eps={optimal_eps:.3f}, min_samples={optimal_min_samples}")
print()

# Perform DBSCAN with optimal parameters
dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
df['cluster'] = dbscan.fit_predict(X_pca)

# Count clusters and noise points
n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
n_noise = list(df['cluster']).count(-1)

# Calculate cluster centers (excluding noise points)
cluster_centers_list = []
cluster_labels = []

for cluster_id in sorted(df['cluster'].unique()):
    if cluster_id != -1:  # Exclude noise
        cluster_mask = df['cluster'] == cluster_id
        cluster_center = X[cluster_mask].mean(axis=0)
        cluster_centers_list.append(cluster_center)
        cluster_labels.append(f'Cluster {cluster_id}')

if cluster_centers_list:
    centers_df = pd.DataFrame(cluster_centers_list, columns=feature_cols)
    centers_df.index = cluster_labels
else:
    centers_df = pd.DataFrame()

# Save results
df_output = df[['match_id', 'home_team', 'away_team', 'cluster']]
df_output = df_output.sort_values(by='cluster')
df_output.to_csv('data/clustered_matches.csv', index=False)

print(f"DBSCAN Clustering Results (eps={optimal_eps:.3f}, min_samples={optimal_min_samples})")
print("=" * 60)
print(f"\nTotal matches: {len(df)}")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise} ({n_noise/len(df)*100:.1f}%)")

if n_clusters > 0:
    print("\nMatches per cluster:")
    print(df['cluster'].value_counts().sort_index())

if not centers_df.empty:
    print("\n" + "=" * 60)
    print("Cluster Characteristics (Cluster Centers):")
    print("=" * 60)
    print(centers_df.round(2))

print("\n" + "=" * 60)
print("Sample matches from each cluster:")
print("=" * 60)

# Show noise points first if any
if n_noise > 0:
    noise_matches = df[df['cluster'] == -1]['match_id'].head(5).tolist()
    print(f"\nNoise points (outliers) - {n_noise} total:")
    for match in noise_matches:
        print(f"  - {match}")

# Show samples from each cluster
for cluster_id in sorted(df['cluster'].unique()):
    if cluster_id != -1:
        cluster_size = len(df[df['cluster'] == cluster_id])
        cluster_matches = df[df['cluster'] == cluster_id]['match_id'].head(5).tolist()
        print(f"\nCluster {cluster_id} ({cluster_size} matches):")
        for match in cluster_matches:
            print(f"  - {match}")

print(f"\n\nResults saved to 'clustered_matches.csv'")
print(f"Parameter exploration plots saved to 'dbscan_parameter_exploration.png'")
print(f"K-distance graph saved to 'k_distance_graph.png'")