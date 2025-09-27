import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create directories
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load data
print("Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

# Save raw data (simulate new data being added later)
df.to_csv("data/housing.csv", index=False)

# Prepare features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Save model
joblib.dump(model, "model/model.pkl")
print("Model saved to model/model.pkl")

# Save metrics for CI checks
with open("model/metrics.txt", "w") as f:
    f.write(f"rmse: {rmse}\n")
    f.write(f"r2: {r2}\n")