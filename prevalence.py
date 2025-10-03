import pandas as pd
import numpy as np

# Read the clustered data
df = pd.read_csv('data/clustered_matches.csv')

# Get all unique teams
all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
all_teams.sort()

# Get all unique clusters
clusters = sorted(df['cluster'].unique())

print("=" * 80)
print("TEAM PREVALENCE ACROSS CLUSTERS")
print("=" * 80)

# Create a summary dataframe
summary_data = []

for team in all_teams:
    row = {'Team': team}
    
    # Count appearances in each cluster (as home or away)
    home_matches = df[df['home_team'] == team]
    away_matches = df[df['away_team'] == team]
    total_matches = len(home_matches) + len(away_matches)
    
    row['Total_Matches'] = total_matches
    
    for cluster in clusters:
        home_in_cluster = len(home_matches[home_matches['cluster'] == cluster])
        away_in_cluster = len(away_matches[away_matches['cluster'] == cluster])
        total_in_cluster = home_in_cluster + away_in_cluster
        
        row[f'Cluster_{cluster}'] = total_in_cluster
        row[f'Cluster_{cluster}_%'] = (total_in_cluster / total_matches * 100) if total_matches > 0 else 0
    
    summary_data.append(row)

# Create dataframe
summary_df = pd.DataFrame(summary_data)

# Sort by total matches (descending)
summary_df = summary_df.sort_values('Total_Matches', ascending=False)

# Display count summary
print("\n1. TEAM APPEARANCE COUNTS BY CLUSTER")
print("-" * 80)
count_cols = ['Team', 'Total_Matches'] + [f'Cluster_{c}' for c in clusters]
print(summary_df[count_cols].to_string(index=False))

# Display percentage summary
print("\n\n2. TEAM APPEARANCE PERCENTAGES BY CLUSTER")
print("-" * 80)
pct_cols = ['Team', 'Total_Matches'] + [f'Cluster_{c}_%' for c in clusters]
pct_df = summary_df[pct_cols].copy()
for col in [f'Cluster_{c}_%' for c in clusters]:
    pct_df[col] = pct_df[col].apply(lambda x: f"{x:.1f}%")
print(pct_df.to_string(index=False))

# Cluster statistics
print("\n\n3. CLUSTER STATISTICS")
print("-" * 80)
for cluster in clusters:
    n_matches = len(df[df['cluster'] == cluster])
    n_teams = len(df[df['cluster'] == cluster]['home_team'].unique()) + \
              len(df[df['cluster'] == cluster]['away_team'].unique())
    n_unique_teams = len(set(df[df['cluster'] == cluster]['home_team'].unique()) | 
                          set(df[df['cluster'] == cluster]['away_team'].unique()))
    
    print(f"\nCluster {cluster}:")
    print(f"  Total matches: {n_matches}")
    print(f"  Unique teams involved: {n_unique_teams}")

# Find teams most associated with each cluster
print("\n\n4. MOST PREVALENT TEAMS PER CLUSTER (Top 5)")
print("-" * 80)
for cluster in clusters:
    print(f"\nCluster {cluster}:")
    cluster_counts = summary_df[['Team', f'Cluster_{cluster}', f'Cluster_{cluster}_%']].copy()
    cluster_counts = cluster_counts[cluster_counts[f'Cluster_{cluster}'] > 0]
    cluster_counts = cluster_counts.sort_values(f'Cluster_{cluster}', ascending=False).head(5)
    
    for _, row in cluster_counts.iterrows():
        print(f"  {row['Team']:20s} - {int(row[f'Cluster_{cluster}'])} matches ({row[f'Cluster_{cluster}_%']:.1f}%)")

# Save to CSV
summary_df.to_csv('data/team_cluster_summary.csv', index=False)
print("\n\n" + "=" * 80)
print("Summary saved to 'team_cluster_summary.csv'")
print("=" * 80)