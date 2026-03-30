from __future__ import annotations

import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_lightgbm_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LGBMRegressor:
    """Train and return a LightGBM regressor."""
    model = LGBMRegressor(
        random_state=random_state,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: LGBMRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Evaluate model and return standard regression metrics."""
    predictions = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_model(model: LGBMRegressor, output_path: str) -> None:
    """Persist model to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path: str) -> LGBMRegressor:
    """Load a persisted model from disk."""
    return joblib.load(model_path)
