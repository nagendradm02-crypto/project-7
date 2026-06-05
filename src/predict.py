# predict.py

import pandas as pd
import joblib

# Load trained model
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

# Make predictions
def predict(model, input_data):

    prediction = model.predict(input_data)

    return prediction

# Main function
def main():

    # Load model
    model_path = "../models/vehicle_model.pkl"
    model = load_model(model_path)

    if model is None:
        return

    # Example input data
    # Replace these values with actual feature values
    sample_data = {
        "Feature1": [10],
        "Feature2": [20],
        "Feature3": [30],
        "Feature4": [40]
    }

    input_df = pd.DataFrame(sample_data)

    # Predict
    prediction = predict(model, input_df)

    print("\nPrediction Result:")
    print(prediction[0])

if __name__ == "__main__":
    main()
