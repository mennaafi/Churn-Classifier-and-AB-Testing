from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error)
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Abstract method to evaluate a model.

        Parameters:
        model (BaseEstimator): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass

class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates a regression model using multiple metrics.

        Returns:
        dict: A dictionary containing MAE, MSE, RMSE, R-Squared, Explained Variance, Median Absolute Error.
        """
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        evs = explained_variance_score(y_test, y_pred)
        median_ae = median_absolute_error(y_test, y_pred)

        metrics = {
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "R-Squared": r2,
            "Explained Variance Score": evs,
            "Median Absolute Error": median_ae,
        }

        return metrics

class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates a classification model using various metrics.

        Returns:
        dict: A dictionary containing Accuracy, Precision, Recall, F1-score, AUC, Confusion Matrix, and ROC Curve.
        """
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        auc = roc_auc_score(y_test, y_pred_prob)

        cm = confusion_matrix(y_test, y_pred)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc,
            "Confusion Matrix": cm,
            "ROC Curve": (fpr, tpr, thresholds)
        }

        return metrics


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific evaluation strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new evaluation strategy.
        """
        self._strategy = strategy

    def evaluate(
        self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Executes evaluation using the current strategy.
        """
        return self._strategy.evaluate_model(model, X_test, y_test)
