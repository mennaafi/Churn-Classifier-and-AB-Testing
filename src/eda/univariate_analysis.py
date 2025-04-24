from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import math
from typing import List


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        pass

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform numerical univariate analysis with:
        1. KDE Plot (Distribution)
        2.  Boxplot
        
        Parameters:
        df (pd.DataFrame): The dataset to analyze.
        feature (str): The numerical feature to analyze.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # KDE Plot
        sns.kdeplot(df[feature].dropna(), ax=axes[0], fill=True, color='skyblue', linewidth=2)
        axes[0].set_title(f'{feature} - KDE Plot', weight="bold", fontsize=15)
        axes[0].set_xlabel(feature, fontsize=12)
        axes[0].set_ylabel("Density", fontsize=12)

        # Boxplot
        sns.boxplot(x=df[feature], ax=axes[1], color='skyblue')
        axes[1].set_title(f'{feature} - Boxplot', weight="bold", fontsize=15)
        axes[1].set_xlabel(feature, fontsize=12)

        plt.tight_layout()
        plt.show()


class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    """
    Performs univariate analysis on categorical features using countplots.

    Parameters:
        df (pd.DataFrame): The dataset to analyze.
        feature (str): The categorical feature to analyze.
       
    """
    def analyze(self, df: pd.DataFrame, features: List[str]):
        if isinstance(features, str):
            features = [features]

        num_features = len(features)
        cols = 2
        rows = math.ceil(num_features / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        if isinstance(axes, plt.Axes):
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]
            count = sns.countplot(x=df[feature], hue=df[feature], palette="Set2", ax=ax, legend=False)

            for p in count.patches:
                count.annotate(f"({int(p.get_height())})",
                               (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                               ha="center", va="bottom", color="black",
                               fontname="monospace", fontsize=10, weight="bold")

            labels = df[feature].value_counts().sort_index().index
            count.set_xlabel("Categories", weight="semibold", fontname="monospace", fontsize=10)
            count.set_ylabel("Count", weight="semibold", fontname="monospace", fontsize=10)
            count.set_xticks(range(len(labels)))
            count.set_xticklabels(labels, fontsize=10, weight="bold")
            count.set_title(f"{feature} - Countplot", weight="bold", fontname="monospace", fontsize=15)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(h_pad=4.0, w_pad=3.0)
        plt.show()

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        self._strategy.analyze(df, feature)
