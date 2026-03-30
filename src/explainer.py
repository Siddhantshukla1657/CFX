from __future__ import annotations

from typing import List

import dice_ml
import pandas as pd
from lightgbm import LGBMRegressor

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN

ACTIONABLE_FEATURES: List[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
]


def build_dice_explainer(
    model: LGBMRegressor,
    train_frame: pd.DataFrame,
    method: str = "random",
) -> dice_ml.Dice:
    """Build and return a configured DiCE explainer."""
    data_object = dice_ml.Data(
        dataframe=train_frame,
        continuous_features=FEATURE_COLUMNS,
        outcome_name=TARGET_COLUMN,
    )
    model_object = dice_ml.Model(model=model, backend="sklearn", model_type="regressor")
    return dice_ml.Dice(data_object, model_object, method=method)


def generate_counterfactuals(
    explainer: dice_ml.Dice,
    query_instance: pd.DataFrame,
    target_min: float,
    target_max: float,
    total_cfs: int = 3,
) -> pd.DataFrame:
    """Generate counterfactuals and return final CF frame."""
    try:
        cf_examples = explainer.generate_counterfactuals(
            query_instance,
            total_CFs=total_cfs,
            desired_range=[target_min, target_max],
            features_to_vary=ACTIONABLE_FEATURES,
        )
    except Exception as e:
        # Return empty DataFrame if DiCE cannot find counterfactuals
        # This can happen if target range is unrealistic or constraints are too restrictive
        print(f"Warning: Could not generate counterfactuals: {str(e)}")
        return pd.DataFrame()

    if not cf_examples.cf_examples_list:
        return pd.DataFrame()

    final_frame = cf_examples.cf_examples_list[0].final_cfs_df
    if final_frame is None:
        return pd.DataFrame()

    return final_frame.copy()
