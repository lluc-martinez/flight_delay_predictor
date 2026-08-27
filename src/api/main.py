from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import mlflow.pyfunc
import os
import warnings
from contextlib import asynccontextmanager

from src.api.schemas import FlightRequest, PredictionResponse
from src.db.database import engine, get_db
from src.db import models

warnings.filterwarnings("ignore", category=UserWarning)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern Lifespan context manager to handle startup and shutdown."""
    global model
    print("Initializing database tables...")
    models.Base.metadata.create_all(bind=engine)
    
    # Use relative path dynamically converted to absolute so it works in Windows and Docker
    model_path = os.path.abspath("production_artifact/model")
    
    if not os.path.exists(model_path):
        print(f"WARNING: Artifact not found at {model_path}. API will return 500 on /predict.")
    else:
        print(f"Loading production model into RAM from {model_path}...")
        model = mlflow.pyfunc.load_model(model_path)
        print("System ready for inference.")
        
    yield  # Yield control to FastAPI

app = FastAPI(
    title="Flight Delay Prediction API",
    description="Production-grade ML Microservice",
    version="1.1.0",
    lifespan=lifespan
)

@app.post("/predict", response_model=PredictionResponse)
def predict_delay(request: FlightRequest, db: Session = Depends(get_db)):
    """Inference endpoint with database auditing."""
    if model is None:
        raise HTTPException(status_code=500, detail="Inference engine offline.")
    
    try:
        input_data = pd.DataFrame([request.model_dump()])
        prediction = model.predict(input_data)
        
        is_delayed = bool(prediction[0])
        probability = 0.99 if is_delayed else 0.01 
        
        audit_record = models.PredictionAudit(
            unique_carrier=request.UniqueCarrier,
            origin=request.Origin,
            dest=request.Dest,
            delay_probability=probability,
            is_delayed=is_delayed
        )
        db.add(audit_record)
        db.commit()
        
        return PredictionResponse(
            delay_probability=probability,
            is_delayed=is_delayed
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")