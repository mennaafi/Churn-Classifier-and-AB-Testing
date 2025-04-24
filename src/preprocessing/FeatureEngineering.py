import pandas as pd
import numpy as np 

class Cleaning:
    @staticmethod
    def drop_features(X: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Drops specified columns from the DataFrame.

        Parameters:
        X : pd.DataFrame
            The DataFrame from which columns will be dropped.
        columns : list
            A list of column names to be dropped from the DataFrame.

        Returns:
        pd.DataFrame
            A new DataFrame with the specified columns dropped.
        
        Notes:
        The `errors='ignore'` argument ensures that if any of the specified columns
        do not exist, the function will not raise an error and will simply skip them.
        """
        X = X.copy()
        return X.drop(columns=columns, errors='ignore')

    @staticmethod
    def convert_columns_dtype(X: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
        """
        Converts the data types of specified columns in the DataFrame.

        Parameters:
        X : pd.DataFrame
            The DataFrame where columns' data types need to be converted.
        dtype_map : dict
            A dictionary where the keys are column names and the values are the 
            target data types ('numeric' or a specific dtype like 'object').

        Returns:
        pd.DataFrame
            A new DataFrame with the columns' data types converted as specified.

        Notes:
        - The 'numeric' dtype will coerce errors to `NaN` (use `errors='coerce'`).
        - The `object` dtype will convert the column to a categorical type.
        """
        X = X.copy()
        for col, dtype in dtype_map.items():
            if col in X.columns:
                if dtype == 'numeric':
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                else:
                    X[col] = X[col].astype(dtype)
        return X
    
    @staticmethod  
    def select_X(df: pd.DataFrame, target_column: str):
        """
        Select feature columns (X) from the DataFrame by excluding the target column.
    
        Parameters:
        - df: pd.DataFrame
        - target_column: str, the name of the target column
    
        Returns:
        - pd.DataFrame: DataFrame with feature columns only
        """
        return df.drop(columns=[target_column])
    
    @staticmethod
    def select_y(df: pd.DataFrame, target_column: str):
        """
        Select the target column (y) from the DataFrame.
    
        Parameters:
        - df: pd.DataFrame
        - target_column: str, the name of the target column
    
        Returns:
        - pd.Series: Target column (y)
        """
        return df[target_column]
    
    @staticmethod
    def replace_values(df: pd.DataFrame, replacements: dict) -> pd.DataFrame:
        """
        Replace specific values in specified columns.

        Parameters:
        - df: pd.DataFrame
        - replacements: dict, where keys are column names and values are 
                    dicts mapping what to replace with what.

        Returns:
        - pd.DataFrame with replaced values
        """
        df = df.copy()
        for col, replace_map in replacements.items():
           if col in df.columns:
              df[col] = df[col].replace(replace_map)
        return df
