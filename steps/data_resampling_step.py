from zenml import step
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from typing import Tuple


@step
def data_resampling_step(X_train_prepared: pd.DataFrame, y_train_prepared: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    ros = RandomOverSampler(sampling_strategy='not majority', random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train_prepared, y_train_prepared.values.ravel())
    
    y_resampled = pd.Series(y_resampled, name='Churn')
    
    return pd.DataFrame(X_resampled, columns=X_train_prepared.columns), y_resampled
