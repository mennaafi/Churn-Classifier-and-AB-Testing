from zenml import step
from src.preprocessing.Pipelines import Pipeline1
import pandas as pd
from typing import Tuple


@step
def data_preprocessing_step(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    pipeline = Pipeline1()
    X_train_prepared, y_train_prepared = pipeline.fit_transform(X_train.copy(), y_train.copy())
    X_test_prepared, y_test_prepared = pipeline.transform(X_test.copy(), y_test.copy())


    if isinstance(y_train_prepared, pd.DataFrame):
        y_train_prepared = y_train_prepared.squeeze()  
    if isinstance(y_test_prepared, pd.DataFrame):
        y_test_prepared = y_test_prepared.squeeze() 

    return X_train_prepared, X_test_prepared, y_train_prepared, y_test_prepared

