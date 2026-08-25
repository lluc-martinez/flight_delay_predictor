import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def build_pipeline() -> Pipeline:
    """Constructs the Scikit-Learn preprocessing and modeling pipeline."""
    # Notice we added 'Total_Origin_Congestion' to the numerical features
    numeric_features = ['Distance', 'Month', 'DayOfWeek', 'CRSDepTime', 'Total_Origin_Congestion']
    numeric_transformer = StandardScaler()

    categorical_features = ['UniqueCarrier', 'Origin', 'Dest']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Using n_jobs=-1 to utilize all CPU cores
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1))
    ])
    
    return pipeline

def main():
    data_path = "data/processed/airlines_features.parquet"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}. Run prepare_data.py first.")

    print("Loading processed data...")
    df = pd.read_parquet(data_path)
    
    # For speed during development, we can sample. Remove this in final training.
    df = df.sample(200000, random_state=42) 
    
    X = df.drop(columns=['Delay'])
    y = df['Delay']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Flight_Delay_Prediction")

    with mlflow.start_run():
        print("Training the model...")
        pipeline.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Metrics -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 15)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, "model")
        print("Model successfully logged to MLflow!")

if __name__ == "__main__":
    main()