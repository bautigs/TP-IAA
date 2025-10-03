import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the clustered data
df = pd.read_csv('data/clustered_matches.csv')

# Prepare data
all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
clusters = sorted(df['cluster'].unique())

# Create summary matrix
team_cluster_matrix = []
team_names = []

for team in sorted(all_teams):
    home_matches = df[df['home_team'] == team]
    away_matches = df[df['away_team'] == team]
    
    row = []
    for cluster in clusters:
        home_in_cluster = len(home_matches[home_matches['cluster'] == cluster])
        away_in_cluster = len(away_matches[away_matches['cluster'] == cluster])
        row.append(home_in_cluster + away_in_cluster)
    
    team_cluster_matrix.append(row)
    team_names.append(team)

matrix = np.array(team_cluster_matrix)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# 1. HEATMAP
fig, ax = plt.subplots(figsize=(10, max(8, len(team_names) * 0.4)))
sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=[f'Cluster {c}' for c in clusters],
            yticklabels=team_names, cbar_kws={'label': 'Number of Matches'},
            linewidths=0.5, ax=ax)
ax.set_title('Team Distribution Across Clusters (Heatmap)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Team', fontsize=12)
plt.tight_layout()
plt.savefig('plots/cluster_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. STACKED BAR CHART (Teams)
fig, ax = plt.subplots(figsize=(14, 8))
x_pos = np.arange(len(team_names))
bottom = np.zeros(len(team_names))

colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))

for i, cluster in enumerate(clusters):
    values = matrix[:, i]
    ax.bar(x_pos, values, bottom=bottom, label=f'Cluster {cluster}', 
           color=colors[i], edgecolor='white', linewidth=1)
    bottom += values

ax.set_xlabel('Team', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
ax.set_title('Team Participation Across Clusters (Stacked Bar)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(team_names, rotation=45, ha='right')
ax.legend(title='Cluster', loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cluster_stacked_bar_teams.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. STACKED BAR CHART (Clusters perspective)
fig, ax = plt.subplots(figsize=(10, 8))
cluster_sizes = matrix.T
x_pos = np.arange(len(clusters))
bottom = np.zeros(len(clusters))

colors = plt.cm.tab20(np.linspace(0, 1, len(team_names)))

for i, team in enumerate(team_names):
    values = cluster_sizes[:, i]
    if values.sum() > 0:  # Only plot teams with matches
        ax.bar(x_pos, values, bottom=bottom, label=team, 
               color=colors[i], edgecolor='white', linewidth=1)
        bottom += values

ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
ax.set_title('Cluster Composition by Team (Stacked Bar)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Cluster {c}' for c in clusters])
ax.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cluster_stacked_bar_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. GROUPED BAR CHART
fig, ax = plt.subplots(figsize=(14, 8))
n_clusters = len(clusters)
bar_width = 0.8 / n_clusters
x_pos = np.arange(len(team_names))

for i, cluster in enumerate(clusters):
    offset = (i - n_clusters/2 + 0.5) * bar_width
    values = matrix[:, i]
    ax.bar(x_pos + offset, values, bar_width, label=f'Cluster {cluster}',
           color=colors[i], edgecolor='white', linewidth=1)

ax.set_xlabel('Team', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Matches', fontsize=12, fontweight='bold')
ax.set_title('Team Distribution Across Clusters (Grouped Bar)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(team_names, rotation=45, ha='right')
ax.legend(title='Cluster', loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cluster_grouped_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. PERCENTAGE STACKED BAR (normalized)
fig, ax = plt.subplots(figsize=(14, 8))
matrix_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
matrix_pct = np.nan_to_num(matrix_pct)  # Handle division by zero

x_pos = np.arange(len(team_names))
bottom = np.zeros(len(team_names))

for i, cluster in enumerate(clusters):
    values = matrix_pct[:, i]
    ax.bar(x_pos, values, bottom=bottom, label=f'Cluster {cluster}',
           color=colors[i], edgecolor='white', linewidth=1)
    bottom += values

ax.set_xlabel('Team', fontsize=12, fontweight='bold')
ax.set_ylabel('Percentage of Matches (%)', fontsize=12, fontweight='bold')
ax.set_title('Team Cluster Distribution (100% Stacked)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(team_names, rotation=45, ha='right')
ax.legend(title='Cluster', loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/cluster_percentage_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. PIE CHARTS (one per cluster)
n_clusters_to_plot = len(clusters)
fig, axes = plt.subplots(1, n_clusters_to_plot, figsize=(6*n_clusters_to_plot, 6))
if n_clusters_to_plot == 1:
    axes = [axes]

for i, cluster in enumerate(clusters):
    cluster_data = matrix[:, i]
    # Only show teams with matches in this cluster
    mask = cluster_data > 0
    labels = [team_names[j] for j in range(len(team_names)) if mask[j]]
    sizes = cluster_data[mask]
    
    if len(sizes) > 0:
        axes[i].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=plt.cm.Set3(np.linspace(0, 1, len(sizes))))
        axes[i].set_title(f'Cluster {cluster} Composition', fontsize=12, fontweight='bold')
    else:
        axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center')
        axes[i].set_title(f'Cluster {cluster} Composition', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/cluster_pie_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print("=" * 80)
print("VISUALIZATIONS CREATED SUCCESSFULLY")
print("=" * 80)
print("\nGenerated files:")
print("  1. cluster_heatmap.png - Color-coded matrix of team-cluster frequencies")
print("  2. cluster_stacked_bar_teams.png - Teams with cluster breakdown")
print("  3. cluster_stacked_bar_clusters.png - Clusters with team composition")
print("  4. cluster_grouped_bar.png - Side-by-side comparison by team")
print("  5. cluster_percentage_stacked.png - Normalized 100% stacked bars")
print("  6. cluster_pie_charts.png - Cluster composition pie charts")
print("\n" + "=" * 80)