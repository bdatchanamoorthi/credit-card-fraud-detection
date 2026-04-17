import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict(data_dict):
    df = pd.DataFrame([data_dict])

    prediction = model.predict(df)[0]

    return "Fraud 🚨" if prediction == 1 else "Legit ✅"