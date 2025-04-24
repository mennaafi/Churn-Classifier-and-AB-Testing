import logging
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from zenml.client import Client
from zenml import ArtifactConfig
from typing import Annotated, Tuple
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model, step

model = Model(
    name="churn_classifier",
    version=None,
    license="MIT",
    description="Customer churn classification model using fine-tuned logistic regression."
)

experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(X_resampled: pd.DataFrame, y_resampled: pd.Series) -> Annotated[LogisticRegression, ArtifactConfig(name="logistic_model", is_model_artifact=True)]:


    """
    Train a fine-tuned logistic regression model on the resampled data.

    Args:
        X_resampled (pd.DataFrame): Resampled feature data.
        y_resampled (pd.DataFrame): Resampled target labels.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """
    if not isinstance(X_resampled, pd.DataFrame):
        raise TypeError("X_resampled must be a pandas DataFrame.")
    if not isinstance(y_resampled, pd.Series):
        raise TypeError("y_resampled must be a Series.")

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        logging.info("Training Logistic Regression model with fine-tuned hyperparameters...")

        best_model = LogisticRegression(
            C=0.1,
            class_weight='balanced',
            penalty='l1',
            solver='liblinear',
            random_state=42
        )

        best_model.fit(X_resampled, y_resampled.values.ravel())
        logging.info("Model training completed successfully.")

        expected_columns = X_resampled.columns.tolist()
        logging.info(f"Model expects the following features: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()

    return best_model


