from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_COLUMN = "MedHouseVal"


def load_california_housing_df() -> pd.DataFrame:
    """Load California Housing dataset as a single DataFrame with target column."""
    housing = fetch_california_housing(as_frame=True)
    frame = housing.frame.copy()
    if TARGET_COLUMN not in frame.columns:
        frame[TARGET_COLUMN] = housing.target
    return frame


def train_test_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Return train/test split and the full frame for reuse across modules."""
    frame = load_california_housing_df()
    x = frame[FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return x_train, x_test, y_train, y_test, frame


def get_feature_bounds(frame: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
    """Return min, max, and median values for each feature."""
    bounds: Dict[str, Tuple[float, float, float]] = {}
    for feature in FEATURE_COLUMNS:
        series = frame[feature]
        bounds[feature] = (float(series.min()), float(series.max()), float(series.median()))
    return bounds
