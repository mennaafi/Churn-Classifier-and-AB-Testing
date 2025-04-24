import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PowerTransformer

class CustomImputer(BaseEstimator, TransformerMixin):
    """
     Custom imputer applies multiple imputation strategies to specified DataFrame columns.

     Supports:
     - median, mean, most_frequent, constant
     - KNN and Iterative imputation with default params

     Parameters:
     - median_cols, mean_cols, freq_cols, constant_cols: Lists of columns for respective strategies
     - knn_cols, iterative_cols: Columns for KNN and Iterative imputers
     - fill_value: Used with constant imputation
    """
    def __init__(self, 
                 median_cols=None,
                 mean_cols=None,
                 freq_cols=None,
                 constant_cols=None,
                 knn_cols=None,
                 iterative_cols=None,
                 fill_value=None,
                 knn_params=None,
                 iterative_params=None):
        
        self.median_cols = median_cols or []
        self.mean_cols = mean_cols or []
        self.freq_cols = freq_cols or []
        self.constant_cols = constant_cols or []
        self.knn_cols = knn_cols or []
        self.iterative_cols = iterative_cols or []
        self.fill_value = fill_value

        self.knn_params = knn_params if knn_params is not None else {'n_neighbors': 3}
        self.iterative_params = iterative_params if iterative_params is not None else {'random_state': 0}

        self.imputers = {}

    def fit(self, X, y=None):
        if self.median_cols:
            self.imputers['median'] = SimpleImputer(strategy='median')
            self.imputers['median'].fit(X[self.median_cols])
        
        if self.mean_cols:
            self.imputers['mean'] = SimpleImputer(strategy='mean')
            self.imputers['mean'].fit(X[self.mean_cols])

        if self.freq_cols:
            self.imputers['most_frequent'] = SimpleImputer(strategy='most_frequent')
            self.imputers['most_frequent'].fit(X[self.freq_cols])

        if self.constant_cols:
            self.imputers['constant'] = SimpleImputer(strategy='constant', fill_value=self.fill_value)
            self.imputers['constant'].fit(X[self.constant_cols])

        if self.knn_cols:
            self.imputers['knn'] = KNNImputer(**self.knn_params)
            self.imputers['knn'].fit(X[self.knn_cols])

        if self.iterative_cols:
            self.imputers['iterative'] = IterativeImputer(**self.iterative_params)
            self.imputers['iterative'].fit(X[self.iterative_cols])
        
        return self

    def transform(self, X):
        X = X.copy()

        if self.median_cols:
            X[self.median_cols] = self.imputers['median'].transform(X[self.median_cols])

        if self.mean_cols:
            X[self.mean_cols] = self.imputers['mean'].transform(X[self.mean_cols])

        if self.freq_cols:
            X[self.freq_cols] = self.imputers['most_frequent'].transform(X[self.freq_cols])

        if self.constant_cols:
            X[self.constant_cols] = self.imputers['constant'].transform(X[self.constant_cols])

        if self.knn_cols:
            X[self.knn_cols] = self.imputers['knn'].transform(X[self.knn_cols])

        if self.iterative_cols:
            X[self.iterative_cols] = self.imputers['iterative'].transform(X[self.iterative_cols])

        return X

class CustomScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler applies the same scaling strategy to specified DataFrame columns.

    Supports:
    - StandardScaler
    - MinMaxScaler
    - RobustScaler

    Parameters:
    - scale_cols: List of columns to scale
    - scaler_type: Type of scaler ('standard', 'minmax', or 'robust')
    """

    def __init__(self, scale_cols=None, scaler_type='standard'):
        self.scale_cols = scale_cols or []
        self.scaler_type = scaler_type

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaler_type must be either 'standard', 'minmax', or 'robust'")

    def fit(self, X, y=None):
        self.scaler.fit(X[self.scale_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.scale_cols] = self.scaler.transform(X[self.scale_cols])
        return X


class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    CustomTransformer applies one of  transformation techniques to selected numeric columns.

    Supports:
    - Logarithmic transformation 
    - Box-Cox transformation 
    - Yeo-Johnson transformation 
    - Winsorization 

    Parameters:
    - method (str): methods to apply ('log', 'box-cox', 'yeo-johnso', or 'winsorize')
    - columns (list): List of column names to transform.
    - winsor_limits (tuple): Tuple (lower, upper) quantiles for clipping in winsorization.

    Use: 
    - 'log': Normalize right-skewed data, Positive-only requirement
    - 'Box-Cox': Normalize right-skewed data, Positive-only requirement
    - 'yeo-johnson': Can handle both left-skewed and right-skewed data, Positive/Negative support
    - 'winsorize': Clip extreme values (outliers) within a specified range, Positive/Negative support
    """

 
    def __init__(self, method='log', columns=None, winsor_limits=(0.01, 0.99)):
        self.method = method
        self.columns = columns or []
        self.winsor_limits = winsor_limits
        self.pt = None  

    def fit(self, X, y=None):
        if self.method in ['yeo-johnson', 'box-cox']:
            self.pt = PowerTransformer(method=self.method, standardize=False)
            self.pt.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if self.method == 'log':
                X[col] = np.log1p(X[col])  
            elif self.method in ['yeo-johnson', 'box-cox']:
                X[self.columns] = self.pt.transform(X[self.columns])
                break  
            elif self.method == 'winsorize':
                lower = X[col].quantile(self.winsor_limits[0])
                upper = X[col].quantile(self.winsor_limits[1])
                X[col] = np.clip(X[col], lower, upper)
            else:
                raise ValueError(f"Unsupported transformation method: {self.method}")
        return X


class CustomEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Encoder that supports multiple encoding methods for categorical variables.

    Supports:
    - OrdinalEncoder
    - OneHotEncoder
    - LabelEncoder

    Parameters:
    - ordinal_cols: List of columns to apply ordinal encoding.
    - onehot_cols: List of columns to apply one-hot encoding.
    - label_cols: List of columns to apply label encoding.
    - ordinal_mapping: Dictionary to map custom ordinal values (not implemented in this version).
    """

    def __init__(self, 
                 ordinal_cols=None, 
                 onehot_cols=None, 
                 label_cols=None, 
                 ordinal_mapping=None):
        
        self.ordinal_cols = ordinal_cols or []
        self.onehot_cols = onehot_cols or []
        self.label_cols = label_cols or []
        self.ordinal_mapping = ordinal_mapping or {}

        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        self.label_encoders = {}

    def fit(self, X, y=None):
        X = X.copy()

        if self.ordinal_cols:
            self.ordinal_encoder.fit(X[self.ordinal_cols])

        if self.onehot_cols:
            self.onehot_encoder.fit(X[self.onehot_cols])

        if self.label_cols:
            for col in self.label_cols:
                le = LabelEncoder()
                le.fit(X[col])
                self.label_encoders[col] = le

        return self

    def transform(self, X):
        X = X.copy()

        if self.ordinal_cols:
            X[self.ordinal_cols] = self.ordinal_encoder.transform(X[self.ordinal_cols])

        if self.onehot_cols:
            onehot_encoded = self.onehot_encoder.transform(X[self.onehot_cols])
            onehot_df = pd.DataFrame(
                onehot_encoded,
                columns=self.onehot_encoder.get_feature_names_out(self.onehot_cols),
                index=X.index
            )
            X = X.drop(columns=self.onehot_cols).join(onehot_df)

        if self.label_cols:
            for col in self.label_cols:
                le = self.label_encoders[col]
                X[col] = X[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        return X
