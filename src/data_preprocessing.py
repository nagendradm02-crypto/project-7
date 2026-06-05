# data_preprocessing.py

import pandas as pd
import numpy as np

# Load dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        return None

# Display dataset information
def dataset_info(df):
    print("\nDataset Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

# Handle missing values
def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values handled.")
    return df

# Remove duplicate rows
def remove_duplicates(df):
    before = df
