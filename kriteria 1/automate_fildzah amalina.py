import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path):
    return pd.read_csv(path)

def preprocess_heart_dataset(df, target_col='target', test_size=0.2, random_state=42):
    # hapus duplikat
    df = df.drop_duplicates()

    # pisah fitur dan target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # kolom numerik & kategorikal
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg',
                        'exang', 'slope', 'ca', 'thal']

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # transform data
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    # ambil nama kolom hasil preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # kembalikan ke DataFrame
    X_train_processed = pd.DataFrame(X_train_arr, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_arr, columns=feature_names)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def main():
    input_path = "kriteria 1/heart deases_raw.csv"
    output_dir = "preprocessing/heart_preprocessing"

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data(input_path)

    # Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_heart_dataset(df)

    # simpan CSV (data siap latih)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    # simpan preprocessor (PKL)
    with open(f"{output_dir}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

 
if __name__ == "__main__":
    main()
