# evaluate.py

import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load dataset
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        return None

# Load trained model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

# Evaluate model
def evaluate_model(model, df):

    # Target column (last column)
    target_column = df.columns[-1]

    X = df.drop(columns=[target_column])
    y_true = df[target_column]

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    print("\nModel Evaluation Results")
    print("-" * 35)
    print(f"Mean Absolute Error (MAE)  : {mae:.4f}")
    print(f"
