import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    df = pd.read_csv("fraudTest.csv")

    
    df = df.drop(columns=["Unnamed: 0", "first", "last", "street"], errors='ignore')

   
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

    
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day

    
    df["age"] = 2024 - df["dob"].dt.year

    
    df = df.drop(columns=["trans_date_trans_time", "dob", "trans_num"])

   
    cat_cols = ["merchant", "category", "gender", "city", "state", "job"]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    
    for col in df.columns:
        if df[col].dtype == "object":
            df = df.drop(columns=[col])

    
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]

   
    X = X.fillna(0)

    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)