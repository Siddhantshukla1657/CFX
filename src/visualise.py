from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from .data_loader import FEATURE_COLUMNS


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
