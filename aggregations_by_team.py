import pandas as pd

def aggregate_team_stats(input_csv_path, output_csv_path):
    """
    Aggregate football match statistics by team, converting home/away metrics 
    to self/rival perspective for each team.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path)
    
    # Initialize list to store all team records
    team_records = []
    
    # Process each match
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Create record for home team (self perspective)
        home_record = {
            'team': home_team,
            'goles_primer_tiempo_self': row['goles_primer_tiempo'],
            'goles_segundo_tiempo_self': row['goles_segundo_tiempo'],
            'cambios_resultado': row['cambios_resultado'],
            'amarillas_total': row['amarillas_total'],
            'rojas_total': row['rojas_total'],
            'goals_self': row['goals_home'],
            'goals_rival': row['goals_away'],
            'avg_posesion_self': row['posesion_home'],
            'avg_posesion_rival': row['posesion_away'],
            'tiros_al_arcototales': row['tiros_al_arcototales'],
            'fouls_total': row['fouls_total'],
            'avg_corners_self': row['corners_home'],
            'avg_corners_rival': row['corners_away'],
            'avg_saves_self': row['saves_home'],
            'avg_saves_rival': row['saves_away']
        }
        
        # Create record for away team (self perspective)
        away_record = {
            'team': away_team,
            'goles_primer_tiempo_self': row['goles_primer_tiempo'],  # This might need clarification
            'goles_segundo_tiempo_self': row['goles_segundo_tiempo'],  # This might need clarification
            'cambios_resultado': row['cambios_resultado'],
            'amarillas_total': row['amarillas_total'],
            'rojas_total': row['rojas_total'],
            'goals_self': row['goals_away'],
            'goals_rival': row['goals_home'],
            'avg_posesion_self': row['posesion_away'],
            'avg_posesion_rival': row['posesion_home'],
            'tiros_al_arcototales': row['tiros_al_arcototales'],
            'fouls_total': row['fouls_total'],
            'avg_corners_self': row['corners_away'],
            'avg_corners_rival': row['corners_home'],
            'avg_saves_self': row['saves_away'],
            'avg_saves_rival': row['saves_home']
        }
        
        team_records.extend([home_record, away_record])
    
    # Convert to DataFrame
    team_df = pd.DataFrame(team_records)
    
    # Group by team and calculate averages
    team_stats = team_df.groupby('team').agg({
        'goles_primer_tiempo_self': 'mean',
        'goles_segundo_tiempo_self': 'mean',
        'cambios_resultado': 'mean',
        'amarillas_total': 'mean',
        'rojas_total': 'mean',
        'goals_self': 'mean',
        'goals_rival': 'mean',
        'avg_posesion_self': 'mean',
        'avg_posesion_rival': 'mean',
        'tiros_al_arcototales': 'mean',
        'fouls_total': 'mean',
        'avg_corners_self': 'mean',
        'avg_corners_rival': 'mean',
        'avg_saves_self': 'mean',
        'avg_saves_rival': 'mean'
    }).round(2)
    
    # Reset index to make team a column
    team_stats = team_stats.reset_index()
    
    # Save to CSV
    team_stats.to_csv(output_csv_path, index=False)
    
    print(f"Team statistics saved to {output_csv_path}")
    print(f"Processed {len(team_stats)} teams")
    print("\nFirst few rows:")
    print(team_stats.head())
    
    return team_stats

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    input_file = "data/first_round.csv"
    output_file = "data/team_averages.csv"
    
    # Run the aggregation
    team_stats = aggregate_team_stats(input_file, output_file)
    
    # Optional: Display some statistics
    print("\nSample team statistics:")
    print(team_stats.to_string(index=False))