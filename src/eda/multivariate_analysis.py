from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        
        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    
    def generate_correlation_heatmap(self, df: pd.DataFrame):

        numeric_df = df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
    
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Blues", linewidths=0.1)
        plt.title("Correlation Heatmap")
        plt.show()


    def generate_pairplot(self, df: pd.DataFrame):

        num_features = df.select_dtypes(include=['number']).columns

        sns.pairplot(df[num_features])
        plt.suptitle("Pair Plot of Numerical Features", y=1.02)
        plt.show()