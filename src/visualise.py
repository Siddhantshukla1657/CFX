from __future__ import annotations

from typing import Tuple

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN


def create_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap of the dataset for identifying redundancy."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('none') # Transparent background for dark theme
    ax.set_facecolor('none')
    
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    
    # Use a diverging palette similar to the dark theme screenshots (Blue to Red)
    cmap = sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(
        corr, cmap="RdBu_r", center=0, vmax=1, vmin=-1,
        square=True, linewidths=0, cbar_kws={"shrink": .75},
        annot=True, fmt=".2f", ax=ax,
        annot_kws={"size": 9}
    )
    
    ax.set_title("Feature Correlation Matrix", color="white", size=14, pad=15)
    
    ax.tick_params(axis='x', colors='white', labelrotation=45)
    ax.tick_params(axis='y', colors='white')
    
    # Adjust colorbar text color
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    fig.tight_layout()
    return fig


def create_distribution_plot(df: pd.DataFrame, feature: str):
    """Create a distribution plot for a specific feature."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=feature, kde=True, color="#0E7490", ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def create_geospatial_plot(df: pd.DataFrame):
    """Create a scatter plot of house values based on coordinates."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df["Longitude"], df["Latitude"], 
        c=df[TARGET_COLUMN], cmap="viridis",
        alpha=0.6, s=df["Population"]/100
    )
    plt.colorbar(scatter, ax=ax, label="House Value (in $100,000s)")
    ax.set_title("Geospatial Distribution of House Prices")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    return fig


def create_shap_summary_plot(
    model: LGBMRegressor,
    x_frame: pd.DataFrame,
    sample_size: int = 500,
):
    """Return a robust SHAP importance bar plot figure for the model."""
    x_sample = x_frame[FEATURE_COLUMNS].sample(
        n=min(sample_size, len(x_frame)),
        random_state=42,
    )

    explainer = shap.Explainer(model)
    shap_values = explainer(x_sample)
    importances = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered_features = np.array(FEATURE_COLUMNS)[order]
    ordered_importances = importances[order]
    ax.barh(ordered_features, ordered_importances, color="#0E7490")
    ax.set_title("SHAP Mean |Value| Feature Importance")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def create_shap_dependence_plot(
    model: LGBMRegressor,
    x_frame: pd.DataFrame,
    feature_name: str,
    sample_size: int = 1000,
):
    """Return a robust SHAP dependence scatter plot figure."""
    x_sample = x_frame[FEATURE_COLUMNS].sample(
        n=min(sample_size, len(x_frame)),
        random_state=42,
    )

    explainer = shap.Explainer(model)
    shap_values = explainer(x_sample)

    if feature_name not in FEATURE_COLUMNS:
        raise ValueError(f"Unknown feature for SHAP dependence plot: {feature_name}")

    feature_index = FEATURE_COLUMNS.index(feature_name)
    feature_values = x_sample[feature_name].to_numpy()
    shap_feature_values = shap_values.values[:, feature_index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(feature_values, shap_feature_values, alpha=0.45, s=20, color="#1D4ED8")
    ax.axhline(0.0, color="#9CA3AF", linewidth=1)
    ax.set_title(f"SHAP Dependence: {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value")
    fig.tight_layout()
    return fig


def build_counterfactual_delta_styler(
    original_row: pd.DataFrame,
    counterfactuals: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.io.formats.style.Styler]:
    """Create a delta DataFrame and a green/red styler against original input."""
    delta = counterfactuals[FEATURE_COLUMNS].subtract(
        original_row[FEATURE_COLUMNS].iloc[0],
        axis=1,
    )

    def _style_value(value: float) -> str:
        if value > 0:
            return "color: #1B8A3C; font-weight: 600;"
        if value < 0:
            return "color: #B42318; font-weight: 600;"
        return "color: #344054;"

    styler = delta.style.format("{:+.3f}").map(_style_value)
    return delta, styler


def create_actual_vs_predicted_plot(y_true: pd.Series, y_pred: np.ndarray):
    """Create a scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    ax.scatter(y_true, y_pred, alpha=0.5, color='#38bdf8', s=15)
    
    # Perfect prediction diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label="Perfect Prediction")
    
    ax.set_title("Actual vs Predicted Prices", color='white', size=14, pad=15)
    ax.set_xlabel("Actual Price ($100k)", color='white')
    ax.set_ylabel("Predicted Price ($100k)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig


def create_residuals_plot(y_true: pd.Series, y_pred: np.ndarray):
    """Create a scatter plot of residuals to check for homoscedasticity."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    ax.scatter(y_pred, residuals, alpha=0.5, color='#f43f5e', s=15)
    ax.axhline(0, color='white', linestyle='--', lw=2, alpha=0.8)
    
    ax.set_title("Residuals Distribution", color='white', size=14, pad=15)
    ax.set_xlabel("Predicted Price ($100k)", color='white')
    ax.set_ylabel("Residuals (Actual - Predicted)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, linestyle='--', alpha=0.2, color='white')
    fig.tight_layout()
    return fig


def create_feature_importance_plot(model: LGBMRegressor):
    """Create a standard LightGBM feature importance bar plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    features = np.array(model.feature_name_)[indices]
    
    # Use a color palette close to the UI theme
    ax.barh(range(len(indices)), importances[indices], color='#0ea5e9', align='center', alpha=0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(features, color='white')
    ax.set_xlabel("Relative Importance (Split)", color='white')
    ax.set_title("LightGBM Native Feature Importance", color='white', size=14, pad=15)
    ax.tick_params(axis='x', colors='white')
    ax.grid(True, axis='x', linestyle='--', alpha=0.2, color='white')
    fig.tight_layout()
    return fig
