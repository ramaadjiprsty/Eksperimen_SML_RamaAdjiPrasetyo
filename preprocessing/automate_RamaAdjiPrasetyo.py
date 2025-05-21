import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

"""
Skrip ini mengotomasi proses preprocessing dataset car evaluation:
- Membaca CSV input
- Menganggap semua fitur adalah kategorikal
- Membangun pipeline preprocessing (imputasi + OneHotEncoder)
- Split data train/test
- Transformasi dan simpan hasil preprocess (train/test) ke folder output

Fungsi utama:
- load_dataset(path)
- build_preprocessor(df, label_column)
- run_preprocessing(input_path, output_dir, label_column, test_size, random_state)

Usage:
$ python automate_RamaAdjiPrasetyo.py --input ../raw_dataset_car_evaluation.csv \
    --output preprocessed_dataset/ \
    --label class --test_size 0.2 --random_state 42
"""


def load_dataset(path: str) -> pd.DataFrame:
    """Membaca dataset CSV dan mengembalikan DataFrame."""
    return pd.read_csv(path)


def build_preprocessor(df: pd.DataFrame, label_column: str):
    """
    Membangun ColumnTransformer untuk preprocessing kategorikal:
    - Imputasi dengan modus
    - One-hot encoding
    """
    categorical_cols = df.drop(columns=[label_column]).select_dtypes(include=['object', 'category']).columns.tolist()

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, categorical_cols)
    ])
    return preprocessor, categorical_cols


def run_preprocessing(input_path: str, output_dir: str, label_column: str, test_size: float = 0.2, random_state: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    df = load_dataset(input_path)
    X = df.drop(columns=[label_column])
    y = df[label_column]

    preprocessor, categorical_cols = build_preprocessor(df, label_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    train_df[label_column] = y_train.reset_index(drop=True)
    test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    test_df[label_column] = y_test.reset_index(drop=True)

    train_path = os.path.join(output_dir, 'train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'test_preprocessed.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Export selesai:\n- {train_path}\n- {test_path}")
    return train_df, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataset car evaluation (from CSV).')
    parser.add_argument('--input', type=str, required=True, help='Path ke input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Folder output untuk simpan hasil')
    parser.add_argument('--label', type=str, default='class', help='Nama kolom target/label')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proporsi data test')
    parser.add_argument('--random_state', type=int, default=42, help='Random state untuk split')
    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input,
        output_dir=args.output,
        label_column=args.label,
        test_size=args.test_size,
        random_state=args.random_state
    )