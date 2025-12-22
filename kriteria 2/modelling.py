import os
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import dagshub


dagshub.init(repo_owner="fildzahfani", repo_name="my-first-repo", mlflow=True)

def load_split_data(data_dir):
    """Load dataset yang sudah dipreprocessing"""
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).iloc[:, 0]
    return X_train, X_test, y_train, y_test

def train_tuning_manual(X_train, X_test, y_train, y_test):
    """Hyperparameter tuning dengan manual logging ke MLflow/DagsHub"""
    mlflow.set_experiment("Heart_Disease_Analysis_DagsHub")

    n_estimators_range = [50, 100, 200]
    max_depth_range = [5, 10, 15]

    best_score = -np.inf
    best_model = None
    best_params = {}

    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            run_name = f"RF_{n_estimators}_{max_depth}"
            with mlflow.start_run(run_name=run_name):
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                # Logging manual ke DagsHub (MLflow)
                mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", report["weighted avg"]["precision"])
                mlflow.log_metric("recall", report["weighted avg"]["recall"])
                mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

                # Simpan model terbaik
                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

                    # Log model dengan artefak input_example
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="best_model",
                        input_example=X_test.iloc[:1]
                    )

                print(f"Run {run_name} → Accuracy: {acc:.4f}")

    return best_model

def main():
    data_dir = "kriteria 1/heart_preprocessing"
    if not os.path.exists(data_dir):
        print(f"Folder '{data_dir}' tidak ditemukan!")
        return

    X_train, X_test, y_train, y_test = load_split_data(data_dir)
    train_tuning_manual(X_train, X_test, y_train, y_test)
    print("\nModelling tuning selesai. Semua tercatat di DagsHub.")

if __name__ == "__main__":
    main()
