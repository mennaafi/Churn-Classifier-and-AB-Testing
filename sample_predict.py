import mlflow
import pandas as pd

mlflow.set_tracking_uri(
    "file:///C:/Users/user/AppData/Roaming/"
    "zenml/local_stores/2c3c38a1-c4b1-4272-8475-82e43fca9ad8/mlruns"
)

RUN_ID = "0bac23d6d671486eab8a0638db25c615"
MODEL_URI = f"runs:/{RUN_ID}/model"

record = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12.0,
    "PhoneService": 1,
    "PaperlessBilling": 1,
    "MonthlyCharges": 79.85,
    "TotalCharges": 950.15,
    "MultipleLines_No phone service": 0.0,
    "MultipleLines_Yes": 1.0,
    "InternetService_Fiber optic": 1.0,
    "InternetService_No": 0.0,
    "OnlineSecurity_No internet service": 0.0,
    "OnlineSecurity_Yes": 1.0,
    "OnlineBackup_No internet service": 0.0,
    "OnlineBackup_Yes": 1.0,
    "DeviceProtection_No internet service": 0.0,
    "DeviceProtection_Yes": 1.0,
    "TechSupport_No internet service": 0.0,
    "TechSupport_Yes": 1.0,
    "StreamingTV_No internet service": 0.0,
    "StreamingTV_Yes": 1.0,
    "StreamingMovies_No internet service": 0.0,
    "StreamingMovies_Yes": 1.0,
    "Contract_One year": 0.0,
    "Contract_Two year": 1.0,
    "PaymentMethod_Credit card (automatic)": 0.0,
    "PaymentMethod_Electronic check": 1.0,
    "PaymentMethod_Mailed check": 0.0,
}

df = pd.DataFrame([record])

int32_cols = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]
df[int32_cols] = df[int32_cols].astype("int32")

model = mlflow.pyfunc.load_model(MODEL_URI)
pred = model.predict(df)

print("Churn prediction:", pred[0])
