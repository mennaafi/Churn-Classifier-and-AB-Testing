import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.preprocessing.FeatureEngineering import Cleaning
from src.preprocessing.Preprocessing import CustomImputer, CustomTransformer, CustomEncoder, CustomScaler

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


class Pipeline1:
    def __init__(self):
        self.target_feature = "Churn"
        self.dropped_features = ['customerID']
        self.replacements = {
            'TotalCharges': {' ': np.nan}
        }
        self.mean_imputed_feature = ['TotalCharges']
        self.dtype_map = {
            'SeniorCitizen': 'object',
            'TotalCharges': 'numeric'
        }
        self.transformed_features = ['TotalCharges']
        self.label_encoded_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        self.onehot_encoded_features = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        self.scaled_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

        self.x_pipeline = Pipeline([
            ('feature_dropper', FunctionTransformer(Cleaning.drop_features, kw_args={'columns': self.dropped_features}, validate=False)),
            ('replace_empty', FunctionTransformer(Cleaning.replace_values, kw_args={'replacements': self.replacements}, validate=False)),
            ('imputer', CustomImputer(mean_cols=self.mean_imputed_feature)),
            ('dtype_converter', FunctionTransformer(Cleaning.convert_columns_dtype, kw_args={'dtype_map': self.dtype_map}, validate=False)),
            ('transformer', CustomTransformer(method='log', columns=self.transformed_features)),
            ('encoder', CustomEncoder(label_cols=self.label_encoded_features, onehot_cols=self.onehot_encoded_features)),
            ('scaler', CustomScaler(scaler_type='standard', scale_cols=self.scaled_features))
        ])

        self.y_pipeline = Pipeline([
            ('encoder', CustomEncoder(label_cols=[self.target_feature]))  
        ])

    def fit_transform(self, X_train, y_train):
        X_train = self.x_pipeline.fit_transform(X_train)

        if isinstance(y_train, pd.Series):
            y_train = y_train.to_frame(name=self.target_feature)

        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train

    def transform(self, X_test, y_test):
        X_test = self.x_pipeline.transform(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.to_frame(name=self.target_feature)

        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test


class Pipeline2:
    def __init__(self):
        self.target_feature = "Churn"
        self.dropped_features = ['customerID']
        self.replacements = {
            'TotalCharges': {' ': np.nan}
        }
        self.mean_imputed_feature = ['TotalCharges']
        self.dtype_map = {
            'SeniorCitizen': 'object',
            'TotalCharges': 'numeric'
        }
        self.transformed_features = ['TotalCharges']
        self.label_encoded_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        self.onehot_encoded_features = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ]
        self.scaled_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.top_selected_features = [
            'TotalCharges', 'MonthlyCharges', 'tenure', 'InternetService_Fiber optic',
            'PaymentMethod_Electronic check', 'Contract_Two year', 'gender',
            'PaperlessBilling', 'OnlineSecurity_Yes', 'Partner', 'TechSupport_Yes',
            'Contract_One year', 'OnlineBackup_Yes', 'SeniorCitizen', 'Dependents',
            'MultipleLines_Yes', 'DeviceProtection_Yes', 'StreamingMovies_Yes',
            'StreamingTV_Yes', 'PaymentMethod_Mailed check'
        ]

        self.x_pipeline = Pipeline([
            ('feature_dropper', FunctionTransformer(Cleaning.drop_features, kw_args={'columns': self.dropped_features}, validate=False)),
            ('replace_empty', FunctionTransformer(Cleaning.replace_values, kw_args={'replacements': self.replacements}, validate=False)),
            ('imputer', CustomImputer(mean_cols=self.mean_imputed_feature)),
            ('dtype_converter', FunctionTransformer(Cleaning.convert_columns_dtype, kw_args={'dtype_map': self.dtype_map}, validate=False)),
            ('transformer', CustomTransformer(method='log', columns=self.transformed_features)),
            ('encoder', CustomEncoder(label_cols=self.label_encoded_features, onehot_cols=self.onehot_encoded_features)),
            ('scaler', CustomScaler(scaler_type='standard', scale_cols=self.scaled_features)),
            ('feature_selector', FunctionTransformer(lambda X: X[self.top_selected_features], validate=False))
        ])

        self.y_pipeline = Pipeline([
            ('encoder', CustomEncoder(label_cols=[self.target_feature])) 
        ])

    def fit_transform(self, X_train, y_train):
        X_train = self.x_pipeline.fit_transform(X_train)

        if isinstance(y_train, pd.Series):
            y_train = y_train.to_frame(name=self.target_feature)

        y_train = self.y_pipeline.fit_transform(y_train)
        return X_train, y_train

    def transform(self, X_test, y_test):
        X_test = self.x_pipeline.transform(X_test)

        if isinstance(y_test, pd.Series):
            y_test = y_test.to_frame(name=self.target_feature)

        y_test = self.y_pipeline.transform(y_test)
        return X_test, y_test
