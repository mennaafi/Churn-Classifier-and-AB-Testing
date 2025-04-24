from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, target: str):
        pass


class BivariateTargetAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, target: str):
        """
        Perform bivariate analysis for all features vs the target column.
        Automatically determines data type (categorical or numerical) and
        uses suitable plots: countplot for categorical, KDE+boxplot for numerical.

        Annotations:
        - Countplot: frequency counts
        - Boxplot: median lines
        - KDE Plot: vertical lines for group means
        """
        features = [col for col in df.columns if col != target]
        cat_features = [col for col in features if df[col].dtype == 'object' or df[col].nunique() < 10]
        num_features = [col for col in features if col not in cat_features]

        # ------- Categorical Features -------
        if cat_features:
            n = len(cat_features)
            rows = math.ceil(n / 2)
            fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows))
            axes = axes.flatten()

            for i, feature in enumerate(cat_features):
                count = sns.countplot(data=df, x=feature, hue=target, palette="Set2", ax=axes[i])
                axes[i].set_title(f'{feature} vs {target}', fontsize=15, weight="bold", fontname="monospace")
                axes[i].tick_params(axis='x', rotation=0)

                for p in count.patches:
                    height = int(p.get_height())
                    count.annotate(f'({height})',
                                   (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                                   ha='center', va='bottom', color='black',
                                   fontname="monospace", fontsize=10, weight="bold")

                count.set_xlabel("Categories", weight="semibold", fontname="monospace", fontsize=10)
                count.set_ylabel("Count", weight="semibold", fontname="monospace", fontsize=10)

                for label in axes[i].get_xticklabels():
                    label.set_fontweight("bold")
                    label.set_fontname("monospace")
                    label.set_fontsize(10)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout(h_pad=4.0, w_pad=3.0)
            plt.show()

        # ------- Numerical Features -------
        if num_features:
            for feature in num_features:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                sns.boxplot(data=df, x=target, y=feature, ax=axes[0])
                axes[0].set_title(f'{feature} by {target} (Boxplot)', fontsize=15, weight="bold", fontname="monospace")

                axes[0].set_xlabel(target, weight="bold", fontname="monospace", fontsize=12)
                axes[0].set_ylabel(feature, weight="bold", fontname="monospace", fontsize=12)

                for label in axes[0].get_xticklabels():
                    label.set_fontweight("bold")
                    label.set_fontname("monospace")
                    label.set_fontsize(10)

                for label in axes[0].get_yticklabels():
                    label.set_fontweight("bold")
                    label.set_fontname("monospace")
                    label.set_fontsize(10)

                medians = df.groupby(target)[feature].median()
                for i, median in enumerate(medians):
                    axes[0].annotate(f'Median: {median:.1f}', 
                                     xy=(i, median),
                                     xytext=(0, 10),
                                     textcoords='offset points',
                                     ha='center', fontsize=9, fontweight='bold', color='black')

                sns.kdeplot(data=df, x=feature, hue=target, fill=True, ax=axes[1], palette='Set2')
                axes[1].set_title(f'{feature} by {target} (KDE)', fontsize=15, weight="bold", fontname="monospace")

                axes[1].set_xlabel(feature, weight="bold", fontname="monospace", fontsize=12)
                axes[1].set_ylabel("Density", weight="bold", fontname="monospace", fontsize=12)

                for label in axes[1].get_xticklabels():
                    label.set_fontweight("bold")
                    label.set_fontname("monospace")
                    label.set_fontsize(10)

                for label in axes[1].get_yticklabels():
                    label.set_fontweight("bold")
                    label.set_fontname("monospace")
                    label.set_fontsize(10)

                for cls in df[target].unique():
                    mean_val = df[df[target] == cls][feature].mean()
                    axes[1].axvline(mean_val, linestyle='--', linewidth=1.5, label=f'{cls} Mean', alpha=0.6)

                plt.suptitle(f'{feature} vs {target}', fontsize=15, fontweight='bold', fontname="monospace")
                plt.tight_layout(h_pad=4.0, w_pad=3.0)
                plt.show()


class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, target: str):
        self._strategy.analyze(df, target)

