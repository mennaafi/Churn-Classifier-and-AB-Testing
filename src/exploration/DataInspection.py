from abc import ABC, abstractmethod
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())


class DataShapeInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Inspects and prints the number of rows and columns in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the shape of the dataframe to the console.
        """
        print(f"\nNumber of rows (observations): {df.shape[0]}")
        print(f"\nNumber of columns (features): {df.shape[1]}")



class DescriptiveStatsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nDescriptive Statistics (Numerical Features):")
        print(df.describe())
        print("\nDescriptive Statistics (Categorical Features):")
        print(df.describe(include=object))

class MissingValueInspectionStrategy(DataInspectionStrategy):

    def inspect(self, df: pd.DataFrame):
        """
        Returns a DataFrame with the number and percentage of null values for each column,
        and displays a heatmap of missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        pd.DataFrame: A DataFrame with two columns: 'Null Counts' and 'Null Percentage %'.
        """
        null_counts = df.isnull().sum()
        null_percentage = (null_counts / len(df)) * 100
        results = pd.DataFrame({
            "Null Counts": null_counts,
            "Null Percentage %": null_percentage
        })
        
        if null_counts.sum() > 0:
            plt.figure(figsize=(6, 6))
            sns.heatmap(df.isna(), cbar=False, cmap="Blues", yticklabels=False)
            plt.title("Missing Value Heatmap")
            plt.show()
        else:
            print("Number of missing values = 0")

        return results


class DuplicateRowInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Prints the number of duplicate rows in the dataset.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None
        """
        print(f"nNumber of Duplicate Rows: {df.duplicated().sum()}")

class UniqueValuesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Prints the number of unique values for each column and the unique values themselves 
        if the number of unique values is less than or equal to 20.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None
        """
        print("\nUnique Values and Their Counts:")
        for column in df.columns:
            num_unique = df[column].nunique() 
            
            print(f"\nColumn: {column}")
            print(f"Number of unique values: {num_unique}")
            
            if num_unique <= 20:
                unique_values = df[column].unique()
                print(f"Unique Values: {unique_values}")

class SkewnessInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Inspects and prints the skewness of numerical columns in the dataframe, separating positive and negative skewness.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the skewness values for each numerical column, separating positive and negative skewness.
        """
        print("\nSkewness of Numerical Columns:")

        numerical_cols = df.select_dtypes(include='number').columns
        
        skewness = df[numerical_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        
        print("\nSkewness Values:")
        print(skewness)

        positive_skew_cols = skewness[skewness > 1].index.tolist()  
        negative_skew_cols = skewness[skewness < -1].index.tolist()  
        
        if positive_skew_cols:
            print("\nHighly Positive Skewed Columns (Skewness > 1):")
            for col in positive_skew_cols:
                print(f"Column: {col}, Skewness: {skewness[col]}")

        if negative_skew_cols:
            print("\nHighly Negative Skewed Columns (Skewness < -1):")
            for col in negative_skew_cols:
                print(f"Column: {col}, Skewness: {skewness[col]}")

        if not positive_skew_cols and not negative_skew_cols:
            print("\nNo highly skewed columns found (Skewness > 1 or < -1).")


class DataInspector:
       def __init__(self, strategy: DataInspectionStrategy):
        self.strategy = strategy

       def set_strategy(self, strategy: DataInspectionStrategy):
           self.strategy = strategy

       def execute_strategy(self, df: pd.DataFrame):
           self.strategy.inspect(df)
