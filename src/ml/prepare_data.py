import pandas as pd
import os

def load_and_clean_data(input_path: str) -> pd.DataFrame:
    """Loads the raw dataset and removes rows with missing target values."""
    print("1. Loading raw dataset...")
    # Read only necessary columns to optimize RAM
    cols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 
            'Origin', 'Dest', 'Distance', 'CRSDepTime', 'CRSArrTime', 'ArrDelay']
    
    df = pd.read_csv(input_path, usecols=cols)
    df = df.dropna(subset=['ArrDelay']).copy()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies domain logic to create targets and congestion features."""
    print("2. Engineering features (Runway Congestion)...")
    
    # Create Binary Target
    df['Delay'] = (df['ArrDelay'] > 15).astype(int)
    
    # Extract Hours
    df['DepHour'] = df['CRSDepTime'] // 100
    df['ArrHour'] = df['CRSArrTime'] // 100
    
    # --- Calculate Congestion ---
    # Departures
    departures = df.groupby(
        ['Year', 'Month', 'DayofMonth', 'Origin', 'DepHour']
    ).size().reset_index(name='Concurrent_Departures')
    
    # Arrivals (mapped back to Origin perspective)
    arrivals = df.groupby(
        ['Year', 'Month', 'DayofMonth', 'Dest', 'ArrHour']
    ).size().reset_index(name='Concurrent_Arrivals')
    
    arrivals = arrivals.rename(columns={'Dest': 'Origin', 'ArrHour': 'DepHour'})
    
    # Merge back to main dataframe
    df = df.merge(departures, on=['Year', 'Month', 'DayofMonth', 'Origin', 'DepHour'], how='left')
    df = df.merge(arrivals, on=['Year', 'Month', 'DayofMonth', 'Origin', 'DepHour'], how='left')
    
    df['Concurrent_Arrivals'] = df['Concurrent_Arrivals'].fillna(0).astype(int)
    df['Total_Origin_Congestion'] = df['Concurrent_Departures'] + df['Concurrent_Arrivals']
    
    # Select final features for the ML model
    final_features = [
        'Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest', 
        'Distance', 'CRSDepTime', 'Total_Origin_Congestion', 'Delay'
    ]
    
    return df[final_features]

def main():
    input_path = "data/raw/DelayedFlights.csv"
    output_path = "data/processed/airlines_features.parquet"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data not found at {input_path}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Execute the pipeline
    df_raw = load_and_clean_data(input_path)
    df_processed = engineer_features(df_raw)
    
    # Save as Parquet for fast reading in the training step
    print(f"3. Saving processed data to {output_path}...")
    df_processed.to_parquet(output_path, index=False)
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()