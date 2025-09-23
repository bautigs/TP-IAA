import pandas as pd

def merge_football_data(stats_file, positions_file, matches_file, output_file):
    """
    Merge three CSV files: team stats, team positions, and matches data.
    
    Args:
        stats_file (str): Path to team statistics CSV
        positions_file (str): Path to team positions text file (one team per line)
        matches_file (str): Path to matches CSV
        output_file (str): Path for output CSV
    """
    
    # Read the team statistics CSV
    team_stats = pd.read_csv(stats_file)
    
    # Read the positions file (simple text file with team names)
    with open(positions_file, 'r', encoding='utf-8') as f:
        positions_list = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create positions dataframe with position numbers (1-indexed)
    positions_df = pd.DataFrame({
        'team': positions_list,
        'position': range(1, len(positions_list) + 1)
    })
    
    # Read matches CSV, keeping only home_team and away_team columns
    matches = pd.read_csv(matches_file)[['home_team', 'away_team']]
    
    # Add home and away positions
    matches = matches.merge(
        positions_df.rename(columns={'team': 'home_team', 'position': 'home_position'}),
        on='home_team',
        how='left'
    )
    
    matches = matches.merge(
        positions_df.rename(columns={'team': 'away_team', 'position': 'away_position'}),
        on='away_team',
        how='left'
    )
    
    # Get all stat columns (excluding the team column)
    stat_columns = [col for col in team_stats.columns if col != 'team']
    
    # Merge home team stats with _home suffix
    home_stats = team_stats.copy()
    home_stats.columns = ['home_team'] + [f"{col}_home" for col in stat_columns]
    
    matches = matches.merge(home_stats, on='home_team', how='left')
    
    # Merge away team stats with _away suffix
    away_stats = team_stats.copy()
    away_stats.columns = ['away_team'] + [f"{col}_away" for col in stat_columns]
    
    matches = matches.merge(away_stats, on='away_team', how='left')
    
    # Save the result
    matches.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")
    
    # Display basic info about the result
    print(f"\nFinal dataset shape: {matches.shape}")
    print(f"Columns: {list(matches.columns)}")
    
    # Show first few rows
    print(f"\nFirst 3 rows:")
    print(matches.head(3))
    
    return matches

# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    stats_file = "data/team_averages.csv"
    positions_file = "data/positions_after_first_round.csv" 
    matches_file = "data/second_round.csv"
    output_file = "data/second_round_with_stats.csv"
    
    try:
        result = merge_football_data(stats_file, positions_file, matches_file, output_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please make sure all input files exist in the correct paths")
    except Exception as e:
        print(f"Error processing files: {e}")