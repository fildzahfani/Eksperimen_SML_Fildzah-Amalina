import os
import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Mengaktifkan autolog (log parameter & metric otomatis)
mlflow.autolog(log_models=False)

def load_split_data(data_dir):
    """Memuat data CSV hasil preprocessing"""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_and_tune(X_train, X_test, y_train, y_test, run_name):
    """Melatih model dengan hyperparameter tuning dan logging ke MLflow"""
    
    mlflow.set_experiment("Heart_Disease_Analysis")
    
    with mlflow.start_run(run_name=run_name) as run:
        # --- Definisikan model dasar ---
        rf = RandomForestClassifier(random_state=42)
        
        # --- Definisikan space hyperparameter untuk tuning ---
        param_dist = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['auto', 'sqrt', 'log2']
        }
        
        # --- Randomized Search ---
        tuner = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1,
            random_state=42
        )
        
        print(f" 🔄 Memulai hyperparameter tuning untuk {run_name}...")
        tuner.fit(X_train, y_train)
        
        best_model = tuner.best_estimator_
        print("✨ Best Parameters:", tuner.best_params_)
        
        # --- Evaluasi ---
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy setelah tuning: {acc:.4f}")
        print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))
        
        # --- Logging Manual ---
        mlflow.sklearn.log_model(best_model, artifact_path="model_random_forest_tuned")
        mlflow.log_metric("accuracy", acc)
        
        return best_model

def parse_args():
    parser = argparse.ArgumentParser(description="Workflow Modelling + Hyperparameter Tuning dengan MLflow")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="preprocessing/heart_preprocessing", 
        help="Path ke folder data preprocessed"
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default="RF_Model_Tuned_v1", 
        help="Nama run untuk tracking MLflow"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f" Folder data '{args.data_dir}' tidak ditemukan!")
        return

    X_train, X_test, y_train, y_test = load_split_data(args.data_dir)
    train_and_tune(X_train, X_test, y_train, y_test, args.run_name)

    print("\nProses modelling & tuning selesai. Jalankan 'mlflow ui' untuk melihat hasil.")

if __name__ == "__main__":
    main()
