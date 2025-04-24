from sklearn.linear_model import LogisticRegression
from zenml import step, Model

@step
def model_loader(model_name: str) -> LogisticRegression:
    """
    Loads the current production model.

    Args:
        model_name: Name and version of the Model to load.

    Returns:
        LogisticRegression: The loaded model.
    """
    model = Model(name=model_name, version="production")
    log_model = model.load_artifact("logistic_model")

    return log_model