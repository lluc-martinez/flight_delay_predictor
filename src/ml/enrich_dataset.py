import pandas as pd
import httpx
import time
import os
from dotenv import load_dotenv

load_dotenv()

AIRPORT_API_URL = "https://api.api-ninjas.com/v1/airports"

def get_airport_coordinates(airport_code: str) -> tuple[float, float] | None:
    """Get latitude and longitude by calling the airport API."""
    headers = {"X-Api-Key": os.getenv("AIRPORT_API_KEY")}
    params = {"iata": airport_code}
    
    try:
        response = httpx.get(AIRPORT_API_URL, headers=headers, params=params, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        if data and isinstance(data, list):
            return float(data[0]["latitude"]), float(data[0]["longitude"])
        return None
    except Exception as e:
        print(f"   [!] Geolocation error for {airport_code}: {e}")
        return None

def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads historical weather matching the EXACT local timezone of the airport."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "wind_speed_10m", "weather_code"],
        "timezone": "auto"
    }
    
    response = httpx.get(url, params=params, timeout=15.0)
    response.raise_for_status()
    data = response.json()["hourly"]
    
    df_weather = pd.DataFrame({
        "datetime": pd.to_datetime(data["time"]),
        "temperature": data["temperature_2m"],
        "wind_speed": data["wind_speed_10m"],
        "weather_code": data["weather_code"]
    })
    
    df_weather["is_raining"] = (df_weather["weather_code"] >= 50).astype(int)
    
    # Extract temporal keys for the exact JOIN
    df_weather["Year"] = df_weather["datetime"].dt.year
    df_weather["Month"] = df_weather["datetime"].dt.month
    df_weather["DayofMonth"] = df_weather["datetime"].dt.day
    df_weather["Hour"] = df_weather["datetime"].dt.hour
    
    return df_weather.drop(columns=["datetime", "weather_code"])

def enrich_flights_data(input_csv="data/raw/DelayedFlights.csv", output_csv="data/processed/enriched_flights.csv"):
    print("1. Loading the original dataset...")
    df_flights = pd.read_csv(input_csv)
    
    # Round the scheduled departure time (e.g., 1830 -> 18)
    df_flights["Hour"] = (df_flights["CRSDepTime"] // 100).astype(int)
    
    # Calculate the date range dynamically from the dataset
    min_year = df_flights["Year"].min()
    max_year = df_flights["Year"].max()
    start_date = f"{min_year}-01-01"
    end_date = f"{max_year}-12-31"
    
    unique_airports = df_flights["Origin"].unique()
    print(f"2. Airports: {len(unique_airports)}. Period: {start_date} to {end_date}.")
    
    weather_dfs = []
    
    for airport in unique_airports:
        print(f"   -> Processing {airport}...")
        coords = get_airport_coordinates(airport)
        
        if coords:
            lat, lon = coords
            try:
                # Use the dynamically calculated dates
                df_w = fetch_historical_weather(lat, lon, start_date, end_date)
                df_w["Origin"] = airport
                weather_dfs.append(df_w)
            except Exception as e:
                print(f"      [!] Error downloading weather for {airport}: {e}")
        time.sleep(1.5) 
            
    df_all_weather = pd.concat(weather_dfs, ignore_index=True)
    
    print("3. Performing the JOIN (relational merge)...")
    # The keys now match the dataset columns exactly
    df_enriched = pd.merge(
        df_flights, 
        df_all_weather, 
        how="left", 
        on=["Origin", "Year", "Month", "DayofMonth", "Hour"]
    )
    
    print("4. Applying fallback values for airports without data...")
    df_enriched["temperature"] = df_enriched["temperature"].fillna(15.0)
    df_enriched["wind_speed"] = df_enriched["wind_speed"].fillna(12.5)
    df_enriched["is_raining"] = df_enriched["is_raining"].fillna(0)
    
    # Remove the temporary hour column created for the merge
    df_enriched = df_enriched.drop(columns=["Hour"])
    
    df_enriched.to_csv(output_csv, index=False)
    print(f"Enriched dataset successfully saved to {output_csv}!")

if __name__ == "__main__":
    enrich_flights_data()