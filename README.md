# ✈️ Flight Delay Prediction API

A production-grade Machine Learning microservice designed to predict flight delays in real-time. This system goes beyond static datasets by dynamically enriching flight inputs with real-time geographical coordinates and weather conditions before inference.

## 🚀 Key Features

* **Asynchronous Feature Engineering:** Intercepts IATA codes and dynamically requests exact geographical coordinates and real-time weather data (wind speed, temperature, rain) via concurrent HTTP calls.
* **Fault-Tolerant Architecture:** Implements strict timeouts and cascading fallbacks. If external APIs (geocoding or weather) fail or throttle, the system gracefully injects baseline metrics to guarantee continuous model inference without dropping user requests.
* **MLOps Pipeline:** Fully versioned model training utilizing MLflow. The Random Forest pipeline integrates `ColumnTransformer` for on-the-fly preprocessing and `class_weight="balanced"` to counteract aviation dataset class imbalance.
* **Containerized Deployment:** Encapsulated via Docker Compose, bridging the FastAPI web server with a relational database (PostgreSQL/SQLite) for prediction auditing.

## 🛠️ Tech Stack

* **API Framework:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-Learn, Pandas, MLflow
* **External Integrations:** `httpx` (Async API calls to API-Ninjas & Open-Meteo)
* **Infrastructure:** Docker, Docker Compose, SQLAlchemy

## 📦 Quick Start

**1. Clone the repository and configure credentials**
Create a `.env` file in the root directory to safely manage the external API key:
`echo "AIRPORT_API_KEY=your_api_ninjas_key" > .env`

**2. Build and launch the infrastructure**
The entire stack (API + Database) is containerized.
`docker-compose up -d --build`

**3. Access the API**
* **Interactive Docs (Swagger UI):** `http://localhost:8000/docs`
* **Health Check:** `http://localhost:8000/`

## 🧠 Inference Example

Send a POST request to `/predict` with the scheduled flight details. The API will silently handle the coordinate resolution and weather extraction.

`curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "UniqueCarrier": "DL",
  "Origin": "ATL",
  "Dest": "JFK",
  "DayOfWeek": 5,
  "Month": 12,
  "Distance": 760,
  "CRSDepTime": 1830,
  "Total_Origin_Congestion": 120
}'`

**Response:**
`{
  "delay_probability": 0.65,
  "is_delayed": true
}`

## 👨‍💻 Author
**Lluc Martinez Busquets**  
*AI Engineer | Computer Engineering*