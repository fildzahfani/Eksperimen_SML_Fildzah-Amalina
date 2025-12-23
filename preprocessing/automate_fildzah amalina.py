import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_heart_dataset(df, target_col='target', test_size=0.2, random_state=42):
    df = df.drop_duplicates().dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg','exang', 'slope', 'ca', 'thal']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    return X_train_df, X_test_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True), preprocessor

def main():
    input_path = "namadataset_raw/heart deases_raw.csv"
    output_path ="preprocessing/automate_fildzah amalina.py"

    base_dir = os.path.dirname(output_path)
    output_dir = os.path.join(base_dir, "heart_preprocessing")
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_heart_dataset(df)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    # Simpan preprocessor
    with open(f"{output_dir}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("Preprocessing selesai & data tersimpan")

if __name__ == "__main__":
    main()
