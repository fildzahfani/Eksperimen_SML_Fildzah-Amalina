import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart_Disease_Basic")
mlflow.sklearn.autolog()

def load_split_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_basic(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="RF_Basic_Model"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print("Accuracy (Basic Model):", acc)

def main():
    data_dir = "preprocessing/heart_preprocessing"
    X_train, X_test, y_train, y_test = load_split_data(data_dir)
    train_basic(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
