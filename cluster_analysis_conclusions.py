import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_clusters():
    """
    Analiza las características principales de cada cluster de partidos.
    """
    
    # Leer datos
    df = pd.read_csv('data/clustered_matches.csv')
    stats_df = pd.read_csv('data/second_round_with_stats.csv')
    merged_df = df.merge(stats_df, on=['home_team', 'away_team'], how='left')
    
    print("=" * 80)
    print("ANÁLISIS DE CARACTERÍSTICAS POR CLUSTER")
    print("=" * 80)
    
    # Almacenar datos para el gráfico
    cluster_data = {}
    
    # Analizar cada cluster
    for cluster in sorted(df['cluster'].unique()):
        cluster_matches = merged_df[merged_df['cluster'] == cluster]
        
        # Calcular 4 características principales
        total_goals = (cluster_matches['goals_self_home'] + cluster_matches['goals_rival_home']).mean()
        possession_diff = abs(cluster_matches['avg_posesion_self_home'] - cluster_matches['avg_posesion_rival_home']).mean()
        total_shots = cluster_matches['tiros_al_arcototales_home'].mean()
        position_diff = abs(cluster_matches['home_position'] - cluster_matches['away_position']).mean()
        
        # Guardar datos
        cluster_data[cluster] = {
            'total_matches': len(cluster_matches),
            'total_goals': total_goals,
            'possession_diff': possession_diff,
            'total_shots': total_shots,
            'position_diff': position_diff
        }
        
        print(f"\nCLUSTER {cluster}")
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

def create_cluster_characteristics_plot(cluster_data):
    """
    Crea un gráfico con las características básicas de cada cluster.
    """
    
    clusters = sorted(cluster_data.keys())
    cluster_labels = [f'Cluster {c}' for c in clusters]
    
    # Extraer datos
    goals = [cluster_data[c]['total_goals'] for c in clusters]
    possession = [cluster_data[c]['possession_diff'] for c in clusters]
    shots = [cluster_data[c]['total_shots'] for c in clusters]
    position = [cluster_data[c]['position_diff'] for c in clusters]
    
    # Crear figura con 4 subgráficos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Características Básicas por Cluster', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # 1. Goles promedio
    ax1 = axes[0, 0]
    bars1 = ax1.bar(cluster_labels, goals, color=colors[:len(clusters)])
    ax1.set_title('Goles Promedio por Partido', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Goles', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, goals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Diferencia de posesión
    ax2 = axes[0, 1]
    bars2 = ax2.bar(cluster_labels, possession, color=colors[:len(clusters)])
    ax2.set_title('Diferencia de Posesión Promedio', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Diferencia (%)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, possession):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Tiros al arco
    ax3 = axes[1, 0]
    bars3 = ax3.bar(cluster_labels, shots, color=colors[:len(clusters)])
    ax3.set_title('Tiros al Arco Promedio', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Tiros', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, shots):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Diferencia de posiciones
    ax4 = axes[1, 1]
    bars4 = ax4.bar(cluster_labels, position, color=colors[:len(clusters)])
    ax4.set_title('Diferencia de Posiciones en Tabla', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Diferencia', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars4, position):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('plots/cluster_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 80)
    print("GRÁFICO GENERADO")
    print("=" * 80)
    print("Archivo guardado: plots/cluster_characteristics.png")
    print("=" * 80)

if __name__ == "__main__":
    print("INICIANDO ANÁLISIS DE CLUSTERS...")
    print()
    
    cluster_data, merged_df = analyze_clusters()
    create_cluster_characteristics_plot(cluster_data)
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
