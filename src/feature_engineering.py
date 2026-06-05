# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load cleaned dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print("Error:", e)
        return None

# Encode categorical columns
def encode_categorical_features(df):
    label_encoders = {}

    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    print("Categorical features encoded.")
    return df, label_encoders

# Create new features
def create_features(df):

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Example Feature 1: Sum of numeric values
    if len(numeric_cols) >= 2:
        df['Total_Feature'] = df[numeric_cols].sum(axis=1)

    # Example Feature 2: Average of numeric values
    if len(numeric_cols) >= 2:
        df['Average_Feature'] = df[numeric_cols].mean(axis=1)

    print("New features created.")
    return df

# Scale numerical features
def scale_features(df):
    scaler = StandardScaler()

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Numerical features scaled.")
    return df, scaler

# Save engineered dataset
def save_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Feature engineered dataset saved to {output_file}")

# Main function
def main():

    input_file = "../data/processed/cleaned_vehicle.csv"
    output_file = "../data/processed/featured_vehicle.csv"

    df = load_data(input_file)

    if df is
