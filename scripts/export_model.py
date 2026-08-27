import os
import shutil
import mlflow

def export_best_model():
    """Simulates a CI/CD pipeline step: extracts the best model for production."""
    print("1. Connecting to MLflow Tracking Server...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    experiment = mlflow.get_experiment_by_name("Flight_Delay_Prediction")
    if not experiment:
        raise ValueError("Experiment not found.")

    print("2. Identifying the best model based on F1 Score...")
    # Fetch runs and sort by F1 score descending
    df_runs = mlflow.search_runs(
        [experiment.experiment_id], 
        order_by=["metrics.f1_score DESC"]
    )
    
    best_run = df_runs.iloc[0]
    print(f"   -> Best Run ID: {best_run.run_id} (F1: {best_run['metrics.f1_score']:.3f})")
    
    model_uri = f"runs:/{best_run.run_id}/model"
    export_path = "production_artifact"
    
    # Clean previous exports
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
        
    print(f"3. Exporting artifact to ./{export_path} ...")
    # This downloads the model artifact out of the messy mlruns folder
    mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=export_path)
    
    print("Success: Model isolated and ready for Docker build.")

if __name__ == "__main__":
    export_best_model()