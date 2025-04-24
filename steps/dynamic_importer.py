import pandas as pd
from zenml import step

@step
def dynamic_importer() -> str:
    """Dynamically imports a small sample of the churn dataset for testing."""
    data = {
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [1, 34],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", "1889.5"],
        "Churn": ["No", "No"]
    }

    df = pd.DataFrame(data)
    json_data = df.to_json(orient="split")

    return json_data