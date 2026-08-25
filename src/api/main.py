from fastapi import FastAPI, HTTPException
import pandas as pd
import mlflow.sklearn
from contextlib import asynccontextmanager
from .schemas import FlightRequest, PredictionResponse
import warnings

# Ignore Scikit-Learn warnings caused by version differences
warnings.filterwarnings("ignore", category=UserWarning)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the latest MLflow model when the server starts."""
    global model
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    try:
        experiment = mlflow.get_experiment_by_name("Flight_Delay_Prediction")
        if not experiment:
            raise ValueError("Experiment not found.")
            
        df_runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
        if df_runs.empty:
            raise ValueError("No trained models found.")
            
        latest_run_id = df_runs.iloc[0].run_id
        model_uri = f"runs:/{latest_run_id}/model"
        
        print(f"Loading model from MLflow (Run ID: {latest_run_id})...")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully into memory!")
        
    except Exception as e:
        print(f"Error loading the model: {e}")

    yield


app = FastAPI(
    title="Flight Delay Prediction API",
    description="ML microservice to predict flight delays",
    version="1.0.0",
    lifespan=lifespan,
)

@app.post("/predict", response_model=PredictionResponse)
def predict_delay(request: FlightRequest):
    """Endpoint that receives flight data and returns a prediction."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on the server.")
    
    # Convert the Pydantic data into a DataFrame that the model can understand
    input_data = pd.DataFrame([request.model_dump()])
    
    # Extract the probability that the class is 1 (Delay)
    probability = model.predict_proba(input_data)[0][1]
    
    return PredictionResponse(
        delay_probability=round(float(probability), 3),
        is_delayed=bool(probability > 0.5)
    )