import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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

def test_predict_valid_flight():
    """Tests that a valid flight payload returns a 200 OK and the correct schema."""
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
    
    # THE FIX: Using the 'with' context manager triggers the lifespan (loads the model)
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "delay_probability" in data
        assert "is_delayed" in data
        assert isinstance(data["delay_probability"], float)
        assert isinstance(data["is_delayed"], bool)

def test_predict_invalid_data():
    """Tests that the API gracefully rejects invalid data (e.g., negative distance)."""
    payload = {
        "UniqueCarrier": "DL",
        "Origin": "ATL",
        "Dest": "JFK",
        "DayOfWeek": 5,
        "Month": 12,
        "Distance": -50,
        "CRSDepTime": 1830,
        "Total_Origin_Congestion": 120
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422