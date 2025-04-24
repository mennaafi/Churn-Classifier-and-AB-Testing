from zenml import step
from sklearn.model_selection import train_test_split
from src.exploration.DataSplitting import DataSplitter, SimpleTrainTestSplitStrategy
import pandas as pd
from typing import Tuple

@step
def data_splitter_step(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    
    X_train = pd.DataFrame(X_train)  
    X_test = pd.DataFrame(X_test)    
    y_train = pd.Series(y_train)     
    y_test = pd.Series(y_test)       
    
    return X_train, X_test, y_train, y_test

