# predict.py
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

# Example prediction: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
example = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])
pred = model.predict(example)
print(f"Predicted House Value: ${pred[0]*100_000:,.2f}")