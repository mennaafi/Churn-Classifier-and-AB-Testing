import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from src.preprocessing.Pipelines import Pipeline1  

@step(enable_cache=False)
def predictor(service: MLFlowDeploymentService, input_data: str) -> np.ndarray:
    """Run an inference request against a prediction service for customer churn.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """

    service.start(timeout=10)

    data = json.loads(input_data)

    data.pop("columns", None)  
    data.pop("index", None)  

    expected_columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn"
    ]

    if len(data["columns"]) != len(expected_columns):
        raise ValueError("The number of columns in the input data does not match the expected columns!")

    for i, column in enumerate(expected_columns):
        if data["columns"][i] != column:
            raise ValueError(f"Expected column '{column}' but found '{data['columns'][i]}' at position {i+1}.")

    df = pd.DataFrame(data["data"], columns=expected_columns)

    pipeline = Pipeline1()
    X_prepared, _ = pipeline.transform(df.copy(), y_test=None)  

    json_list = json.loads(json.dumps(list(X_prepared.T.to_dict().values())))
    data_array = np.array(json_list)

    prediction = service.predict(data_array)

    return prediction
