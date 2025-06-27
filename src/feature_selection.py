import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from tueplots.constants.color import rgb


def calculate_correlation_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Computes Pearson correlation scores between each feature in X and a target y.

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Target variable.

    Returns:
        pd.Series: Pearson correlation scores for each feature with the target,
                   sorted by absolute correlation value in descending order.
    """

    print(f"Calculating Pearson correlation for {X.shape[1]} features with the target...")
    correlation_scores = X.apply(lambda feature_column: feature_column.corr(y, method='pearson'))

    correlation_scores_series = pd.Series(
        correlation_scores, index=X.columns, name="PearsonCorrelation"
    )
    
    # Sort by absolute value to see strongest correlations (positive or negative)
    temp_df = pd.DataFrame({'corr': correlation_scores, 'abs_corr': correlation_scores.abs()})
    temp_df = temp_df.sort_values(by='abs_corr', ascending=False)
    correlation_scores_series = temp_df['corr']
        
    return correlation_scores_series


def calculate_mutual_information_scores(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> pd.Series:
    """
    Computes mutual information scores between features in X and a target y.

    Args:
        X (pd.DataFrame): DataFrame of features.
        y (pd.Series): Target variable (Assumed to be continuous).
        random_state (int): Random state for reproducibility.

    Returns:
        pd.Series: Mutual information scores for each feature, sorted in descending order.
    """
    print(f"Calculating Mutual Information scores for {X.shape[1]} features...")

    y_array = y.values.ravel()

    # assumes y is continuous
    mi_scores = mutual_info_regression(X, y_array, random_state=random_state)

    mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MutualInfoScore")
    mi_scores_series = mi_scores_series.sort_values(ascending=False)

    return mi_scores_series


def plot_feature_metrics_summary(
    correlation_scores: pd.Series,
    mi_scores: pd.Series,
    target_variable_name: str,
    top_n: int = 20,
) -> None:
    """
    Plots Pearson Correlation and Mutual Information scores for features against a target
    in a single figure with two subplots.

    Args:
        correlation_scores (pd.Series): Pearson correlation scores (features vs. target).
        mi_scores (pd.Series): Mutual Information scores (features vs. target).
        target_variable_name (str): Name of the target variable for plot titles.
        top_n (int, optional): Number of top features to display in each plot. Defaults to 20.
        figure_size (tuple, optional): Size of the overall figure. Defaults to (12, 10).
    """

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Feature Importance with Target: {target_variable_name}", fontsize=16)

    # Plot Pearson Correlation Scores
    scores_to_plot_corr = correlation_scores.head(top_n)
    # For correlation, we care about magnitude and sign, so sort by absolute for top_n, then plot actual values
    # The sorting in calculate_target_correlation_scores already handles sorting by absolute value.
    scores_to_plot_corr[::-1].plot(kind='barh', ax=axes[0], color=rgb.tue_blue)
    axes[0].set_title(f"Top {top_n} Features by Pearson Correlation (Ranked by Absolute Value)")
    axes[0].set_xlabel("Pearson Correlation Coefficient")
    axes[0].set_ylabel("Feature")

    # Plot Mutual Information Scores
    scores_to_plot_mi = mi_scores.head(top_n)
    scores_to_plot_mi[::-1].plot(kind='barh', ax=axes[1], color=rgb.tue_red)
    axes[1].set_title(f"Top {top_n} Features by Mutual Information")
    axes[1].set_xlabel("Mutual Information Score")
    axes[1].set_ylabel("Feature")

    plt.tight_layout()
    plt.show()


def create_correlation_matrix(
    df: pd.DataFrame, title: str = "Feature Correlation Matrix"
) -> pd.DataFrame:
    """
    Computes and plots the Pearson correlation matrix for a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame for which to compute the correlation matrix.
        title (str): Title for the heatmap plot.

    Returns:
        pd.DataFrame: The computed correlation matrix.
    """
    print(f"Calculating Pearson correlation matrix for {df.shape[1]} features...")
    correlation_matrix = df.corr(method="pearson")

    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return correlation_matrix

def create_mutual_information_matrix(
    df: pd.DataFrame, title: str = "Feature Mutual Information Matrix"
) -> pd.DataFrame:
    """
    Computes and plots the Mutual Information matrix for a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame for which to compute the Mutual Information matrix.
        title (str): Title for the heatmap plot.

    Returns:
        pd.DataFrame: The computed Mutual Information matrix.
    """
    print(f"Calculating Mutual Information matrix for {df.shape[1]} features...")
    mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            if i == j:
                mi_matrix.iloc[i, j] = 1.0
            else:
                mi_matrix.iloc[i, j] = mutual_info_regression(
                    df.iloc[:, [i]], df.iloc[:, j]
                )[0]

    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        mi_matrix.astype(float), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return mi_matrix
