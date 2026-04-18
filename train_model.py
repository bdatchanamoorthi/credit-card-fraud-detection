from preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


X_train, X_test, y_train, y_test = load_and_preprocess()


model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Performance:\n")
print(classification_report(y_test, y_pred))


joblib.dump(model, "model.pkl")

print("✅ Model saved successfully")