import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    df = pd.read_csv("fraudTest.csv")

    # Drop useless columns
    df = df.drop(columns=["Unnamed: 0", "first", "last", "street"], errors='ignore')

    # Fix date parsing (IMPORTANT FIX)
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], 
        dayfirst=True, 
        errors='coerce'
    )

    df["dob"] = pd.to_datetime(
        df["dob"], 
        dayfirst=True, 
        errors='coerce'
    )

    # Extract useful time features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day

    # Convert DOB → age
    df["age"] = 2024 - df["dob"].dt.year

    # Drop original columns
    df = df.drop(columns=["trans_date_trans_time", "dob", "trans_num"])

    # Encode ONLY required categorical columns
    cat_cols = ["merchant", "category", "gender", "city", "state", "job"]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Final check: remove any remaining non-numeric columns
    for col in df.columns:
        if df[col].dtype == "object":
            df = df.drop(columns=[col])

    # Split features and target
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

    # Fill missing values (important safety)
    X = X.fillna(0)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)