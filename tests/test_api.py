import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from src.api.main import app
from src.db.database import get_db, Base

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_audit.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

# --- THE TESTS ---

# 1. Interceptamos la función antes de que se ejecute en el código real
@patch("src.api.main.get_weather_features")
def test_predict_valid_flight(mock_get_weather):
    """Prueba que el payload válido devuelve 200 OK con clima simulado."""
    
    # 2. Obligamos al mock a devolver estos datos instantáneamente (sin usar internet)
    mock_get_weather.return_value = {
        "wind_speed": 10.0,
        "temperature": 22.5,
        "is_raining": 0
    }
    
    payload = {
        "UniqueCarrier": "DL",
        "Origin": "ATL",
        "Dest": "JFK",
        "DayOfWeek": 5,
        "Month": 12,
        "Distance": 760,
        "CRSDepTime": 1830,
        "Total_Origin_Congestion": 120
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "delay_probability" in data
        assert "is_delayed" in data
        
        # 3. Verificamos que nuestro código intentó buscar el clima de Atlanta (ATL)
        mock_get_weather.assert_called_once_with("ATL")

@patch("src.api.main.get_weather_features")
def test_predict_invalid_data(mock_get_weather):
    """Prueba que los errores de Pydantic bloquean la petición antes de buscar el clima."""
    payload = {
        "UniqueCarrier": "DL",
        "Origin": "ATL",
        "Dest": "JFK",
        "DayOfWeek": 5,
        "Month": 12,
        "Distance": -50,  # Inválido: Pydantic espera >= 0
        "CRSDepTime": 1830,
        "Total_Origin_Congestion": 120
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        
        # 4. Pydantic debería haber abortado la petición antes de llamar al clima
        mock_get_weather.assert_not_called()