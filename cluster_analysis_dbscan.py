import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_clusters_dbscan():
    """
    Analiza las características principales de cada cluster de partidos para DBSCAN.
    """
    
    # Leer datos de DBSCAN (usar 0.9 PCA que tiene cluster -1)
    df = pd.read_csv('data/clustered_matches_dbscan_0_9_PCA.csv')
    stats_df = pd.read_csv('data/second_round_with_stats.csv')
    merged_df = df.merge(stats_df, on=['home_team', 'away_team'], how='left')
    
    # Incluir todos los clusters, incluyendo ruido (-1)
    all_clusters = sorted(df['cluster'].unique())
    
    print("=" * 80)
    print("ANÁLISIS DE CARACTERÍSTICAS POR CLUSTER - DBSCAN")
    print("=" * 80)
    
    # Almacenar datos para el gráfico
    cluster_data = {}
    
    # Analizar cada cluster, incluyendo ruido (-1)
    for cluster in all_clusters:
        cluster_matches = merged_df[merged_df['cluster'] == cluster]
        
        # Calcular 4 características principales
        total_goals = (cluster_matches['goals_self_home'] + cluster_matches['goals_rival_home']).mean()
        possession_diff = abs(cluster_matches['avg_posesion_self_home'] - cluster_matches['avg_posesion_rival_home']).mean()
        total_shots = cluster_matches['tiros_al_arcototales_home'].mean()
        # Diferencia de posiciones en la tabla (igual que KMeans)
        position_diff = abs(cluster_matches['home_position'] - cluster_matches['away_position']).mean()
        
        # Guardar datos
        cluster_data[cluster] = {
            'total_matches': len(cluster_matches),
            'total_goals': total_goals,
            'possession_diff': possession_diff,
            'total_shots': total_shots,
            'position_diff': position_diff
        }
        
        # Etiqueta para el cluster
        if cluster == -1:
            cluster_label = "RUIDO (Outliers)"
        else:
            cluster_label = f"CLUSTER {cluster}"
        
        print(f"\n{cluster_label}")
        print("-" * 60)
        print(f"Total de partidos: {len(cluster_matches)}")
        print()
        
        # Característica 1: Goles promedio
        print(f"1. Goles promedio por partido: {total_goals:.2f}")
        if total_goals >= 3.0:
            print(f"   Partidos con alta producción ofensiva")
        elif total_goals >= 2.0:
            print(f"   Partidos con producción ofensiva moderada")
        else:
            print(f"   Partidos con baja producción ofensiva")
        print()
        
        # Característica 2: Diferencia de posesión
        print(f"2. Diferencia de posesión promedio: {possession_diff:.2f}%")
        if possession_diff >= 15:
            print(f"   Partidos con dominio claro de un equipo")
        elif possession_diff >= 8:
            print(f"   Partidos con ligera ventaja en posesión")
        else:
            print(f"   Partidos equilibrados en posesión")
        print()
        
        # Característica 3: Tiros totales
        print(f"3. Tiros al arco promedio: {total_shots:.2f}")
        if total_shots >= 15:
            print(f"   Partidos con alta intensidad ofensiva")
        elif total_shots >= 10:
            print(f"   Partidos con intensidad ofensiva media")
        else:
            print(f"   Partidos con baja intensidad ofensiva")
        print()
        
        # Característica 4: Diferencia de posiciones
        print(f"4. Diferencia de posiciones en la tabla: {position_diff:.2f}")
        if position_diff >= 10:
            print(f"   Partidos entre equipos de diferentes niveles")
        elif position_diff >= 5:
            print(f"   Partidos con cierta diferencia de nivel")
        else:
            print(f"   Partidos entre equipos de nivel similar")
        
        print()
    
    return cluster_data, merged_df

def create_individual_characteristic_plots_dbscan(cluster_data):
    """
    Crea gráficos individuales para cada característica en una carpeta separada para DBSCAN.
    """
    
    # Crear directorio para los gráficos individuales de DBSCAN
    os.makedirs('plots/caracteristicas_individuales', exist_ok=True)
    
    clusters = sorted(cluster_data.keys())
    # Crear etiquetas, mostrando "Ruido" para cluster -1
    cluster_labels = [f'Ruido' if c == -1 else f'Cluster {c}' for c in clusters]
    
    # Extraer datos
    goals = [cluster_data[c]['total_goals'] for c in clusters]
    possession_diff = [cluster_data[c]['possession_diff'] for c in clusters]
    shots = [cluster_data[c]['total_shots'] for c in clusters]
    position_diff = [cluster_data[c]['position_diff'] for c in clusters]
    
    # Colores: usar gris para ruido, otros colores para clusters normales
    colors = []
    for c in clusters:
        if c == -1:
            colors.append('#808080')  # Gris para ruido
        else:
            colors.append(['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][c % 5])
    
    # 1. Gráfico de Goles promedio
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cluster_labels, goals, color=colors[:len(clusters)])
    ax.set_title('Goles Promedio por Partido - DBSCAN', fontweight='bold', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Goles', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, goals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/caracteristicas_individuales/1_goles_promedio_dbscan.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Gráfico de Diferencia de posesión
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cluster_labels, possession_diff, color=colors[:len(clusters)])
    ax.set_title('Diferencia de Posesión Promedio - DBSCAN', fontweight='bold', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Diferencia (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, possession_diff):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/caracteristicas_individuales/2_diferencia_posesion_dbscan.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gráfico de Tiros al arco
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cluster_labels, shots, color=colors[:len(clusters)])
    ax.set_title('Tiros al Arco Promedio - DBSCAN', fontweight='bold', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Tiros', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, shots):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/caracteristicas_individuales/3_tiros_al_arco_dbscan.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gráfico de Diferencia de posiciones
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cluster_labels, position_diff, color=colors[:len(clusters)])
    ax.set_title('Diferencia de Posiciones en Tabla - DBSCAN', fontweight='bold', fontsize=14)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Diferencia', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, position_diff):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/caracteristicas_individuales/4_diferencia_posiciones_dbscan.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 80)
    print("GRÁFICOS INDIVIDUALES GENERADOS - DBSCAN")
    print("=" * 80)
    print("Carpeta: plots/caracteristicas_individuales/")
    print("  - 1_goles_promedio_dbscan.png")
    print("  - 2_diferencia_posesion_dbscan.png")
    print("  - 3_tiros_al_arco_dbscan.png")
    print("  - 4_diferencia_posiciones_dbscan.png")
    print("=" * 80)

if __name__ == "__main__":
    print("INICIANDO ANÁLISIS DE CLUSTERS - DBSCAN...")
    print()
    
    cluster_data, merged_df = analyze_clusters_dbscan()
    create_individual_characteristic_plots_dbscan(cluster_data)
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO - DBSCAN")
    print("=" * 80)

