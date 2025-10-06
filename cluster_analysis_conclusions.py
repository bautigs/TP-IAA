import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_cluster_quality():
    """
    Analiza la calidad de cada cluster basándose en estadísticas de entretenimiento
    y determina cuáles son los mejores partidos para ver.
    """
    
    # Leer datos
    df = pd.read_csv('data/clustered_matches.csv')
    stats_df = pd.read_csv('data/second_round_with_stats.csv')
    merged_df = df.merge(stats_df, on=['home_team', 'away_team'], how='left')
    
    print("=" * 80)
    print("ANÁLISIS DE CALIDAD DE PARTIDOS POR CLUSTER")
    print("=" * 80)
    
    # Definir métricas de entretenimiento
    entertainment_metrics = {
        'total_goals': 'goals_self_home + goals_rival_home',
        'possession_difference': 'abs(avg_posesion_self_home - avg_posesion_rival_home)',
        'shot_intensity': 'tiros_al_arcototales_home',
        'physicality': 'fouls_total_home',
        'drama': 'cambios_resultado_home',
        'competitiveness': 'abs(home_position - away_position)'
    }
    
    cluster_analysis = {}
    
    for cluster in sorted(df['cluster'].unique()):
        cluster_matches = merged_df[merged_df['cluster'] == cluster]
        
        # Calcular métricas de entretenimiento
        analysis = {
            'total_matches': len(cluster_matches),
            'avg_total_goals': (cluster_matches['goals_self_home'] + cluster_matches['goals_rival_home']).mean(),
            'avg_possession_diff': abs(cluster_matches['avg_posesion_self_home'] - cluster_matches['avg_posesion_rival_home']).mean(),
            'avg_shots': cluster_matches['tiros_al_arcototales_home'].mean(),
            'avg_fouls': cluster_matches['fouls_total_home'].mean(),
            'avg_drama': cluster_matches['cambios_resultado_home'].mean(),
            'avg_competitiveness': abs(cluster_matches['home_position'] - cluster_matches['away_position']).mean(),
            'home_win_rate': (cluster_matches['goals_self_home'] > cluster_matches['goals_rival_home']).mean(),
            'draw_rate': (cluster_matches['goals_self_home'] == cluster_matches['goals_rival_home']).mean()
        }
        
        cluster_analysis[cluster] = analysis
    
    # Crear DataFrame para análisis
    analysis_df = pd.DataFrame(cluster_analysis).T
    
    # Calcular score de entretenimiento (0-100)
    # Normalizar cada métrica y ponderar
    weights = {
        'avg_total_goals': 0.25,      # Más goles = más entretenido
        'avg_possession_diff': -0.15, # Menos diferencia = más equilibrado
        'avg_shots': 0.20,            # Más tiros = más acción
        'avg_fouls': 0.10,            # Más faltas = más físico
        'avg_drama': 0.20,            # Más cambios = más emocionante
        'avg_competitiveness': -0.10  # Menos diferencia = más competitivo
    }
    
    # Normalizar métricas (0-1)
    for metric in weights.keys():
        min_val = analysis_df[metric].min()
        max_val = analysis_df[metric].max()
        analysis_df[f'{metric}_normalized'] = (analysis_df[metric] - min_val) / (max_val - min_val)
    
    # Calcular score de entretenimiento
    analysis_df['entertainment_score'] = 0
    for metric, weight in weights.items():
        if weight > 0:
            analysis_df['entertainment_score'] += analysis_df[f'{metric}_normalized'] * weight
        else:
            analysis_df['entertainment_score'] += (1 - analysis_df[f'{metric}_normalized']) * abs(weight)
    
    # Convertir a escala 0-100
    analysis_df['entertainment_score'] = analysis_df['entertainment_score'] * 100
    
    # Ordenar por score de entretenimiento
    analysis_df = analysis_df.sort_values('entertainment_score', ascending=False)
    
    print("\nRANKING DE CLUSTERS POR ENTERTENIMIENTO:")
    print("-" * 60)
    for i, (cluster, row) in enumerate(analysis_df.iterrows()):
        print(f"{i+1}. CLUSTER {cluster}: {row['entertainment_score']:.1f}/100")
        print(f"   - Goles promedio: {row['avg_total_goals']:.2f}")
        print(f"   - Cambios de resultado: {row['avg_drama']:.2f}")
        print(f"   - Competitividad: {row['avg_competitiveness']:.2f}")
        print(f"   - Tiros promedio: {row['avg_shots']:.2f}")
        print()
    
    return analysis_df, merged_df

def get_best_matches_to_watch():
    """
    Identifica los mejores partidos específicos para ver basándose en el análisis.
    """
    
    analysis_df, merged_df = analyze_cluster_quality()
    
    print("=" * 80)
    print("MEJORES PARTIDOS PARA VER POR CLUSTER")
    print("=" * 80)
    
    # Obtener el mejor cluster
    best_cluster = analysis_df.index[0]
    best_cluster_matches = merged_df[merged_df['cluster'] == best_cluster]
    
    print(f"\nMEJOR CLUSTER: {best_cluster}")
    print(f"Score de entretenimiento: {analysis_df.loc[best_cluster, 'entertainment_score']:.1f}/100")
    
    # Mostrar los mejores partidos del cluster ganador
    print(f"\nMEJORES PARTIDOS DEL CLUSTER {best_cluster}:")
    print("-" * 50)
    
    # Ordenar por total de goles (más entretenidos)
    best_matches = best_cluster_matches.copy()
    best_matches['total_goals'] = best_matches['goals_self_home'] + best_matches['goals_rival_home']
    best_matches = best_matches.sort_values('total_goals', ascending=False)
    
    for i, (_, match) in enumerate(best_matches.head(10).iterrows()):
        total_goals = match['goals_self_home'] + match['goals_rival_home']
        drama = match['cambios_resultado_home']
        print(f"{i+1:2d}. {match['home_team']} vs {match['away_team']}")
        print(f"    Goles: {total_goals:.0f} | Drama: {drama:.1f} | Tiros: {match['tiros_al_arcototales_home']:.0f}")
    
    return best_cluster, best_cluster_matches

def generate_insights_and_recommendations():
    """
    Genera insights y recomendaciones basadas en el análisis de clusters.
    """
    
    analysis_df, merged_df = analyze_cluster_quality()
    
    print("\n" + "=" * 80)
    print("INSIGHTS Y RECOMENDACIONES")
    print("=" * 80)
    
    # Análisis del mejor cluster
    best_cluster = analysis_df.index[0]
    best_data = analysis_df.loc[best_cluster]
    
    print(f"\nCLUSTER RECOMENDADO: {best_cluster}")
    print(f"Razones para elegir este cluster:")
    print(f"• Score de entretenimiento: {best_data['entertainment_score']:.1f}/100")
    print(f"• Goles promedio por partido: {best_data['avg_total_goals']:.2f}")
    print(f"• Cambios de resultado promedio: {best_data['avg_drama']:.2f}")
    print(f"• Competitividad promedio: {best_data['avg_competitiveness']:.2f}")
    
    # Análisis por tipo de espectador
    print(f"\nRECOMENDACIONES POR TIPO DE ESPECTADOR:")
    print("-" * 50)
    
    print(f"\nPara espectadores que buscan EMOCION:")
    emotion_cluster = analysis_df.sort_values('avg_drama', ascending=False).index[0]
    emotion_data = analysis_df.loc[emotion_cluster]
    print(f"• Cluster {emotion_cluster}: {emotion_data['avg_drama']:.2f} cambios de resultado promedio")
    
    print(f"\nPara espectadores que buscan GOLES:")
    goals_cluster = analysis_df.sort_values('avg_total_goals', ascending=False).index[0]
    goals_data = analysis_df.loc[goals_cluster]
    print(f"• Cluster {goals_cluster}: {goals_data['avg_total_goals']:.2f} goles promedio")
    
    print(f"\nPara espectadores que buscan COMPETITIVIDAD:")
    comp_cluster = analysis_df.sort_values('avg_competitiveness', ascending=True).index[0]
    comp_data = analysis_df.loc[comp_cluster]
    print(f"• Cluster {comp_cluster}: {comp_data['avg_competitiveness']:.2f} diferencia de posición promedio")
    
    print(f"\nPara espectadores que buscan CALIDAD TECNICA:")
    tech_cluster = analysis_df.sort_values('avg_possession_diff', ascending=True).index[0]
    tech_data = analysis_df.loc[tech_cluster]
    print(f"• Cluster {tech_cluster}: {tech_data['avg_possession_diff']:.2f} diferencia de posesión promedio")
    
    # Estadísticas generales
    print(f"\nESTADISTICAS GENERALES:")
    print("-" * 30)
    print(f"• Total de partidos analizados: {len(merged_df)}")
    print(f"• Promedio de goles por partido: {(merged_df['goals_self_home'].sum() + merged_df['goals_rival_home'].sum()) / len(merged_df):.2f}")
    print(f"• Promedio de cambios de resultado: {merged_df['cambios_resultado_home'].mean():.2f}")
    print(f"• Promedio de tiros por partido: {merged_df['tiros_al_arcototales_home'].mean():.2f}")
    
    # Recomendaciones específicas
    print(f"\nRECOMENDACIONES ESPECIFICAS:")
    print("-" * 35)
    print(f"1. Para maxima emocion: Ve partidos del Cluster {emotion_cluster}")
    print(f"2. Para ver goles: Ve partidos del Cluster {goals_cluster}")
    print(f"3. Para partidos equilibrados: Ve partidos del Cluster {comp_cluster}")
    print(f"4. Para calidad tecnica: Ve partidos del Cluster {tech_cluster}")
    print(f"5. Para el mejor balance general: Ve partidos del Cluster {best_cluster}")
    
    return analysis_df

def create_entertainment_visualization():
    """
    Crea visualizaciones del análisis de entretenimiento.
    """
    
    analysis_df, _ = analyze_cluster_quality()
    
    # Configurar estilo
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Entretenimiento por Cluster', fontsize=16, fontweight='bold')
    
    # 1. Score de entretenimiento
    ax1 = axes[0, 0]
    clusters = [f'Cluster {i}' for i in analysis_df.index]
    scores = analysis_df['entertainment_score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax1.bar(clusters, scores, color=colors)
    ax1.set_title('Score de Entretenimiento por Cluster', fontweight='bold')
    ax1.set_ylabel('Score (0-100)')
    ax1.set_ylim(0, 100)
    
    # Añadir valores en las barras
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Goles promedio
    ax2 = axes[0, 1]
    ax2.bar(clusters, analysis_df['avg_total_goals'], color=colors)
    ax2.set_title('Goles Promedio por Partido', fontweight='bold')
    ax2.set_ylabel('Goles')
    
    # 3. Cambios de resultado
    ax3 = axes[1, 0]
    ax3.bar(clusters, analysis_df['avg_drama'], color=colors)
    ax3.set_title('Cambios de Resultado Promedio', fontweight='bold')
    ax3.set_ylabel('Cambios')
    
    # 4. Competitividad
    ax4 = axes[1, 1]
    ax4.bar(clusters, analysis_df['avg_competitiveness'], color=colors)
    ax4.set_title('Competitividad Promedio', fontweight='bold')
    ax4.set_ylabel('Diferencia de Posición')
    
    plt.tight_layout()
    plt.savefig('plots/cluster_entertainment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizacion guardada en: plots/cluster_entertainment_analysis.png")

if __name__ == "__main__":
    print("INICIANDO ANÁLISIS DE CALIDAD DE PARTIDOS...")
    
    # Ejecutar análisis completo
    analysis_df = generate_insights_and_recommendations()
    best_cluster, best_matches = get_best_matches_to_watch()
    create_entertainment_visualization()
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    print(f"✅ Mejor cluster identificado: {best_cluster}")
    print(f"✅ Insights generados")
    print(f"✅ Recomendaciones creadas")
    print(f"✅ Visualización guardada")
    print("=" * 80)
