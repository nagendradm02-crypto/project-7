# create_model.py

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Sample training data
X = np.array([
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [20, 30, 40, 50],
    [25, 35, 45, 55]
])

y = np.array([100, 150, 200, 250])

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "vehicle_model.pkl")

print("vehicle_model.pkl created successfully!")
