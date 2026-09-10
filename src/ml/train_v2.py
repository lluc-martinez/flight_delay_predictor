import pandas as pd
import mlflow.sklearn
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def train_and_export_v2():
    print("1. Loading enriched historical dataset...")
    data = pd.read_csv("data/processed/enriched_flights.csv")
    
    # Create the binary target variable based on ArrDelay (arrival delay)
    if 'delay' not in data.columns:
        data['delay'] = (data['ArrDelay'] > 15).astype(int)
        
    # Remove columns that will not be available at inference time
    cols_to_drop = ["delay", "ArrDelay", "DepDelay", "ActualElapsedTime", "AirTime", "TaxiIn", "TaxiOut", "Cancelled", "CancellationCode", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "FlightNum", "TailNum"]
    X = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    y = data["delay"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("2. Building the Scikit-Learn V2 pipeline...")
    categorical_features = ["UniqueCarrier", "Origin", "Dest"]
    # Select the numeric features that the API will receive
    numeric_features = ["Distance", "CRSDepTime", "wind_speed", "temperature", "is_raining"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), [col for col in numeric_features if col in X.columns]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [col for col in categorical_features if col in X.columns])
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced"))
    ])

    print("3. Training and logging to MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Flight_Delay_Prediction")
    
    with mlflow.start_run(run_name="Real_Weather_Enriched_RF") as run:
        pipeline.fit(X_train, y_train)
        
        # Absolute predictions (0 or 1) and probabilities (for ROC-AUC)
        preds = pipeline.predict(X_test)
        preds_proba = pipeline.predict_proba(X_test)[:, 1] 
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, preds_proba)
        }
        
        # Log all metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")
        
        model_uri = f"runs:/{run.info.run_id}/model"
        
        # Print results to the console for quick validation
        print("\n   [V2 Model Test Results]")
        print(f"   -> Accuracy: {metrics['accuracy']:.3f}")
        print(f"   -> Precision: {metrics['precision']:.3f}")
        print(f"   -> Recall: {metrics['recall']:.3f}")
        print(f"   -> F1 Score: {metrics['f1_score']:.3f}")
        print(f"   -> ROC AUC: {metrics['roc_auc']:.3f}\n")

    print("4. Promoting the new model to production...")
    export_path = "production_artifact"
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=export_path)
    
    print("V2 model exported successfully with real weather support!")

if __name__ == "__main__":
    train_and_export_v2()