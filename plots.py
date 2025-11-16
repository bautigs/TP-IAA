import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Read the clustered data
df = pd.read_csv('data/clustered_matches_dbscan_0_9_PCA.csv')

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

# Función para obtener etiqueta de cluster
def get_cluster_label(cluster):
    if cluster == -1:
        return 'Ruido'
    else:
        return f'Cluster {cluster}'

# 1. HEATMAP
fig, ax = plt.subplots(figsize=(10, max(8, len(team_names) * 0.4)))
sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=[get_cluster_label(c) for c in clusters],
            yticklabels=team_names, cbar_kws={'label': 'Número de Partidos'},
            linewidths=0.5, ax=ax)
ax.set_title('Distribución de Equipos por Cluster (Mapa de Calor)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Equipo', fontsize=12)
plt.tight_layout()
plt.savefig('plots/clustering_nuevo/cluster_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. STACKED BAR CHART (Teams)
fig, ax = plt.subplots(figsize=(14, 8))
x_pos = np.arange(len(team_names))
bottom = np.zeros(len(team_names))

colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))

for i, cluster in enumerate(clusters):
    values = matrix[:, i]
    ax.bar(x_pos, values, bottom=bottom, label=get_cluster_label(cluster), 
           color=colors[i], edgecolor='white', linewidth=1)
    bottom += values

ax.set_xlabel('Equipo', fontsize=12, fontweight='bold')
ax.set_ylabel('Número de Partidos', fontsize=12, fontweight='bold')
ax.set_title('Participación de Equipos por Cluster (Barras Apiladas)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(team_names, rotation=45, ha='right')
ax.legend(title='Cluster', loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/clustering_nuevo/cluster_stacked_bar_teams.png', dpi=300, bbox_inches='tight')
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
ax.set_ylabel('Número de Partidos', fontsize=12, fontweight='bold')
ax.set_title('Composición de Clusters por Equipo (Barras Apiladas)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels([get_cluster_label(c) for c in clusters])
ax.legend(title='Equipo', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/clustering_nuevo/cluster_stacked_bar_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. GROUPED BAR CHART
fig, ax = plt.subplots(figsize=(14, 8))
n_clusters = len(clusters)
bar_width = 0.8 / n_clusters
x_pos = np.arange(len(team_names))

for i, cluster in enumerate(clusters):
    offset = (i - n_clusters/2 + 0.5) * bar_width
    values = matrix[:, i]
    ax.bar(x_pos + offset, values, bar_width, label=get_cluster_label(cluster),
           color=colors[i], edgecolor='white', linewidth=1)

ax.set_xlabel('Equipo', fontsize=12, fontweight='bold')
ax.set_ylabel('Número de Partidos', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Equipos por Cluster (Barras Agrupadas)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(team_names, rotation=45, ha='right')
ax.legend(title='Cluster', loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/clustering_nuevo/cluster_grouped_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. PERCENTAGE STACKED BAR (normalized)
fig, ax = plt.subplots(figsize=(14, 8))
matrix_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100
matrix_pct = np.nan_to_num(matrix_pct)  # Handle division by zero

x_pos = np.arange(len(team_names))
bottom = np.zeros(len(team_names))

for i, cluster in enumerate(clusters):
    values = matrix_pct[:, i]
    ax.bar(x_pos, values, bottom=bottom, label=get_cluster_label(cluster),
           color=colors[i], edgecolor='white', linewidth=1)
    bottom += values

ax.set_xlabel('Equipo', fontsize=12, fontweight='bold')
ax.set_ylabel('Porcentaje de Partidos (%)', fontsize=12, fontweight='bold')
ax.set_title('Distribución de Clusters por Equipo (100% Apilado)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(team_names, rotation=45, ha='right')
ax.legend(title='Cluster', loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/clustering_nuevo/cluster_percentage_stacked.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. PIE CHARTS (one per cluster) - Combined view
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
        axes[i].set_title(f'Composición {get_cluster_label(cluster)}', fontsize=12, fontweight='bold')
    else:
        axes[i].text(0.5, 0.5, 'Sin Datos', ha='center', va='center')
        axes[i].set_title(f'Composición {get_cluster_label(cluster)}', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/clustering_nuevo/cluster_pie_charts.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. INDIVIDUAL PIE CHARTS (one file per cluster)
# Create directory for individual cluster plots
os.makedirs('plots/clustering_nuevo/clusters_individuales', exist_ok=True)

for i, cluster in enumerate(clusters):
    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_data = matrix[:, i]  # Usar índice i en lugar de cluster
    
    # Only show teams with matches in this cluster
    mask = cluster_data > 0
    labels = [team_names[j] for j in range(len(team_names)) if mask[j]]
    sizes = cluster_data[mask]
    
    if len(sizes) > 0:
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                           startangle=90,
                                           colors=plt.cm.Set3(np.linspace(0, 1, len(sizes))),
                                           textprops={'fontsize': 10})
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        cluster_label = get_cluster_label(cluster)
        ax.set_title(f'Composición {cluster_label}\n({int(cluster_data.sum())} partidos)', 
                    fontsize=14, fontweight='bold', pad=20)
    else:
        cluster_label = get_cluster_label(cluster)
        ax.text(0.5, 0.5, 'Sin Datos', ha='center', va='center', fontsize=16)
        ax.set_title(f'Composición {cluster_label}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    # Usar el valor del cluster para el nombre del archivo (puede ser -1)
    plt.savefig(f'plots/clustering_nuevo/clusters_individuales/cluster_{cluster}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("=" * 80)
print("VISUALIZACIONES CREADAS EXITOSAMENTE")
print("=" * 80)
print("\nArchivos generados:")
print("  1. cluster_heatmap.png - Matriz de frecuencias equipo-cluster")
print("  2. cluster_stacked_bar_teams.png - Equipos con desglose por cluster")
print("  3. cluster_stacked_bar_clusters.png - Clusters con composición de equipos")
print("  4. cluster_grouped_bar.png - Comparación lado a lado por equipo")
print("  5. cluster_percentage_stacked.png - Barras apiladas 100% normalizadas")
print("  6. cluster_pie_charts.png - Gráficos de torta de composición de clusters")
print("\nCarpeta 'clusters_individuales' con gráficos por cluster:")
for cluster in clusters:
    print(f"  - cluster_{cluster}.png")
print("\n" + "=" * 80)
