import pandas as pd
import ast
import numpy as np

class MinutoPartido:
    def __init__(self, minuto: str):
        minuto = minuto.replace("'", "").replace('"', '').strip()
        if "+" in minuto:
            self.minuto = int(minuto.split("+")[0])
            self.adicional = int(minuto.split("+")[1])
        else:
            self.minuto = int(minuto)
            self.adicional = 0
        if self.minuto <= 45:
            self.periodo = 1
        else:
            self.periodo = 2
    
    def __str__(self):
        return f"{self.minuto}+{self.adicional} (Periodo {self.periodo})"
      
    def __gt__(self, other):
        if self.minuto == other.minuto:
            return self.adicional > other.adicional
        return self.minuto > other.minuto

def match_result_to_total_goals(result: str) -> int:
    
    away_goals, home_goals = result.strip().split('-')
    return int(away_goals) + int(home_goals)

def cambio_resultados(home_team_goals_current_time: str, away_team_goals_current_time: str) -> int:
    # Handle NaN or empty values
    if pd.isna(home_team_goals_current_time) or home_team_goals_current_time == '' or home_team_goals_current_time == '[]':
        home_team_goals_current_time = []
    else:
        home_team_goals_current_time = ast.literal_eval(home_team_goals_current_time) if isinstance(home_team_goals_current_time, str) else home_team_goals_current_time
    
    if pd.isna(away_team_goals_current_time) or away_team_goals_current_time == '' or away_team_goals_current_time == '[]':
        away_team_goals_current_time = []
    else:
        away_team_goals_current_time = ast.literal_eval(away_team_goals_current_time) if isinstance(away_team_goals_current_time, str) else away_team_goals_current_time
    
    # If no goals, no changes
    if not home_team_goals_current_time and not away_team_goals_current_time:
        return 0
    
    away_goals = [(MinutoPartido(str(i)), "away") for i in away_team_goals_current_time]
    home_goals = [(MinutoPartido(str(i)), "home") for i in home_team_goals_current_time]

    all_goals = away_goals + home_goals
    all_goals.sort(key=lambda x: x[0])
    
    diferencia = 0
    anterior = 0
    cambios = 0
    
    for _, equipo in all_goals:
        anterior = diferencia
        
        if equipo == "home":
            diferencia += 1
        else:
            diferencia -= 1
            
        if anterior in [0, 1, -1] and diferencia in [0, 1, -1]:
            cambios += 1
            
    return cambios

def convert_date(date: str) -> pd.Timestamp:
    "orignal format: DD.MM"
    day, month = date.split(".")
    if month == "1":
        month = "10"
    day, month = int(day), int(month)
    if month < 7:  # assuming season starts in August and ends in May
        year = 2023
    else:
        year = 2022
    return pd.Timestamp(year=year, month=month, day=day)

def safe_int_convert(value):
    """Safely convert value to int, handling NaN and string representations"""
    if pd.isna(value) or value == '' or value == 'nan':
        return 0
    try:
        if isinstance(value, str):
            # Handle list-like strings
            if value.startswith('[') and value.endswith(']'):
                parsed = ast.literal_eval(value)
                return len(parsed) if isinstance(parsed, list) else 0
        return int(float(value))
    except (ValueError, TypeError, SyntaxError):
        return 0

def safe_float_convert(value):
    """Safely convert value to float, handling NaN and string representations"""
    if pd.isna(value) or value == '' or value == 'nan':
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def count_list_items(value):
    """Count items in a list-like string or return 0"""
    if pd.isna(value) or value == '' or value == '[]':
        return 0
    try:
        if isinstance(value, str) and value.startswith('['):
            parsed = ast.literal_eval(value)
            return len(parsed) if isinstance(parsed, list) else 0
        elif isinstance(value, list):
            return len(value)
        else:
            return 0
    except (ValueError, SyntaxError):
        return 0

def process_football_data(input_file: str, output_file: str = None):
    """
    Process football match data and extract required statistics
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file (optional)
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file, dtype={"Date_day": str})
    
    # Filter for Premier League 2022/2023 season only
    print(f"Original dataset: {len(df)} matches")
    df = df[(df['League'] == 'Premier-league') & (df['season_year'] == '2022/2023')]
    print(f"After filtering for Premier League 2022/2023: {len(df)} matches")
    
    if len(df) == 0:
        print("Warning: No matches found for Premier League 2022/2023 season.")
        print("Please check the values in 'Lig' and 'season_year' columns.")
        return pd.DataFrame()
    
    # Create a new dataframe with ONLY the required statistics (drop all other columns)
    result_df = pd.DataFrame()
    
    # Basic match information
    result_df['home_team'] = df['home_team']
    result_df['away_team'] = df['away_team']
    result_df["date"] = df["Date_day"].apply(convert_date)
    
    
    
    # First and second half goals (convert to int)
    result_df['goles_primer_tiempo'] = df['first_half'].apply(match_result_to_total_goals)
    result_df['goles_segundo_tiempo'] = df['second_half'].apply(match_result_to_total_goals)
    
    # Result changes during the match
    result_df['cambios_resultado'] = df.apply(
        lambda row: cambio_resultados(
            row.get('home_team_goals_current_time', '[]'), 
            row.get('away_team_goals_current_time', '[]')
        ), axis=1
    )
    
    # Cards
    result_df['amarillas_total'] = (
        df['home_team_yellow_card'].apply(count_list_items) + 
        df['away_team_yellow_card'].apply(count_list_items)
    )
    
    result_df['rojas_total'] = (
        df['home_team_red_card'].apply(count_list_items) + 
        df['away_team_red_card'].apply(count_list_items)
    )
    
    result_df["goals_home"] = df["home_team_goals"].apply(count_list_items)
    result_df["goals_away"] = df["away_team_goals"].apply(count_list_items)
    
    
    convert_possession = lambda x: safe_float_convert(x.replace('%', '').strip())
    
    result_df['posesion_home'] = df['Ball_Possession_Home'].apply(convert_possession)
    result_df['posesion_away'] = df['Ball_Possession_Host'].apply(convert_possession)
    
    # Total goal attempts (chances de gol total)
    result_df['tiros_al_arcototales'] = (
        df['Shots_on_Goal_Host'].apply(safe_int_convert) + 
        df['Shots_on_Goal_Home'].apply(safe_int_convert)
    )
    
    # Total fouls
    result_df['fouls_total'] = (
        df['Fouls_Home'].apply(safe_int_convert) + 
        df['Fouls_Host'].apply(safe_int_convert)
    )
    
    # Corner kicks
    result_df['corners_home'] = df['Corner_Kicks_Home'].apply(safe_int_convert)
    result_df['corners_away'] = df['Corner_Kicks_Host'].apply(safe_int_convert)
    
    
    
    
    # Goalkeeper saves
    result_df['saves_home'] = df['Goalkeeper_Saves_Home'].apply(safe_int_convert)
    result_df['saves_away'] = df['Goalkeeper_Saves_Host'].apply(safe_int_convert)
    
    #sort by date 
    result_df = result_df.sort_values(by="date").reset_index(drop=True)
    
    # Save to file if output_file is specified
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Replace 'input.csv' with your actual input file path
    # Replace 'output.csv' with your desired output file path
    
    input_file = 'data/Football.csv'  # Your input CSV file
    output_file = 'data/processed_football_stats.csv'  # Output file
    
    try:
        processed_data = process_football_data(input_file, output_file)
        
        if len(processed_data) == 0:
            print("No data to process after filtering.")
            
        print("Data processing completed successfully!")
        print(f"Processed {len(processed_data)} Premier League 2022/2023 matches")
        print(f"\nColumns in output: {list(processed_data.columns)}")
        print("\nFirst few rows of processed data:")
        print(processed_data.head())
        
        # Display some basic statistics
        print(f"\nBasic statistics:")
        print(f"Average total goal chances per match: {processed_data['tiros_al_arcototales'].mean():.2f}")
        print(f"Average result changes per match: {processed_data['cambios_resultado'].mean():.2f}")
        print(f"Average total cards per match: {(processed_data['amarillas_total'] + processed_data['rojas_total']).mean():.2f}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your input data format and try again.")