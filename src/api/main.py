from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
import pandas as pd
import mlflow.pyfunc
import httpx
import os
import warnings

from src.api.schemas import FlightRequest, PredictionResponse
from src.db.database import engine, get_db
from src.db import models

warnings.filterwarnings("ignore", category=UserWarning)

# External endpoints
AIRPORT_API_URL = "https://api.api-ninjas.com/v1/airports"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
model = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Initialize the database and load the model into memory."""
    global model
    print("Initializing database tables...")
    models.Base.metadata.create_all(bind=engine)
    
    model_path = os.path.abspath("production_artifact/model")
    if not os.path.exists(model_path):
        print(f"WARNING: Artifact not found at {model_path}.")
    else:
        print(f"Loading production model into RAM from {model_path}...")
        model = mlflow.pyfunc.load_model(model_path)
        
    yield

app = FastAPI(
    title="Flight Delay Prediction API",
    description="Production-grade ML Microservice with Dynamic Geocoding and Weather Enrichment",
    version="3.0.0",
    lifespan=lifespan
)

async def get_airport_coordinates(airport_code: str) -> tuple[float, float] | None:
    """Look up the coordinates (lat, lon) for an IATA code."""
    headers = {"X-Api-Key": os.getenv("AIRPORT_API_KEY", "DUMMY_KEY")} # Configure this in .env or Docker
    params = {"iata": airport_code}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(AIRPORT_API_URL, headers=headers, params=params, timeout=1.5)
            response.raise_for_status()
            data = response.json()
            
            if data and isinstance(data, list):
                return float(data[0]["latitude"]), float(data[0]["longitude"])
            return None
            
    except (httpx.TimeoutException, httpx.RequestError, IndexError, KeyError) as e:
        print(f"Airport API failed for {airport_code} ({e}).")
        return None

async def get_weather_features(airport_code: str) -> dict:
    """Chain geolocation and weather lookup with fault tolerance."""
    baseline_weather = {"wind_speed": 12.5, "temperature": 15.0, "is_raining": 0}
    
    # 1. Try to retrieve coordinates
    coords = await get_airport_coordinates(airport_code)
    if not coords:
        print(f"Cascading fallback: Missing coordinates for {airport_code}; skipping Weather API.")
        return baseline_weather
        
    # 2. If coordinates are available, request the weather
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "current_weather": True
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(WEATHER_API_URL, params=params, timeout=1.5)
            response.raise_for_status()
            data = response.json().get("current_weather", {})
            
            return {
                "wind_speed": data.get("windspeed", 12.5),
                "temperature": data.get("temperature", 15.0),
                "is_raining": int(data.get("weathercode", 0) > 50)
            }
            
    except (httpx.TimeoutException, httpx.RequestError) as e:
        print(f"Weather API failed for {airport_code} ({e}). Using baseline fallback.")
        return baseline_weather

@app.post("/predict", response_model=PredictionResponse)
async def predict_delay(request: FlightRequest, db: Session = Depends(get_db)):
    """Run inference with real-time feature engineering and fault tolerance."""
    if model is None:
        raise HTTPException(status_code=500, detail="Inference engine offline.")
    
    try:
        # 1. Asynchronous feature engineering
        weather_features = await get_weather_features(request.Origin)
        
        # 2. Assemble input
        raw_input = request.model_dump()
        raw_input.update(weather_features)
        input_data = pd.DataFrame([raw_input])
        
        # 3. Inference
        prediction = model.predict(input_data)
        is_delayed = bool(prediction[0])
        probability = 0.99 if is_delayed else 0.01 
        
        # 4. Audit
        audit_record = models.PredictionAudit(
            unique_carrier=request.UniqueCarrier,
            origin=request.Origin,
            dest=request.Dest,
            delay_probability=probability,
            is_delayed=is_delayed
        )
        db.add(audit_record)
        db.commit()
        
        return PredictionResponse(delay_probability=probability, is_delayed=is_delayed)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")