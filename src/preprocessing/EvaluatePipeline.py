import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

class EvaluatePipeline:
    def __init__(self, X_train_transformed, y_train_transformed):
        """
        Initializes the EvaluatePipeline class with transformed data.

        :param X_train_transformed: The features of the transformed training data
        :param y_train_transformed: The target variable of the transformed training data
        """
        self.X_train = X_train_transformed
        self.y_train = y_train_transformed

        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs'),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "CalibratedClassifierCV": CalibratedClassifierCV(base_estimator=LogisticRegression(max_iter=1000), cv=3),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42)
        }

        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.scorers = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }

    def evaluate(self):
        """
        Evaluates the selected models using cross-validation on the transformed data.

        :return: A DataFrame containing evaluation metrics for the models
        """
        results = {}

        for model_name, model in self.models.items():
            model_results = {}
            for metric, scorer in self.scorers.items():
                try:
                    score = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring=scorer).mean()
                except Exception as e:
                    score = None  
                model_results[metric] = score

            results[model_name] = model_results

        results_df = pd.DataFrame.from_dict(results, orient='index')
        return results_df
