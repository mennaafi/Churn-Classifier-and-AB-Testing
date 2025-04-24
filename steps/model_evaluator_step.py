import pandas as pd
from sklearn.linear_model import LogisticRegression
from zenml import step
from typing import Dict, Tuple
import logging
from src.preprocessing.EvaluteModel import ModelEvaluator, ClassificationModelEvaluationStrategy

@step
def model_evaluator_step(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the trained Logistic Regression model and return classification metrics.

    Args:
        model (LogisticRegression): Trained classifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas Series.")

    evaluator = ModelEvaluator(strategy=ClassificationModelEvaluationStrategy())

    evaluation_metrics = evaluator.evaluate(model, X_test, y_test)

    logging.info("Classification model evaluation completed.")
    logging.info(f"Evaluation metrics: {evaluation_metrics}")

    return evaluation_metrics
