# tests/test_model.py
import joblib
import pandas as pd
from sklearn.metrics import r2_score

def test_r2_threshold():
    # Load model and data
    model = joblib.load("model/model.pkl")
    df = pd.read_csv("data/housing.csv")
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    
    # Predict
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Assert model is decent
    assert r2 > 0.5, f"R² too low: {r2:.2f}"
    print(f"✅ Model passed R² test: {r2:.2f}")