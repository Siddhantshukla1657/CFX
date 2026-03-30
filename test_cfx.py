from __future__ import annotations

from src.data_loader import FEATURE_COLUMNS, load_california_housing_df
from src.explainer import build_dice_explainer, generate_counterfactuals
from src.model import load_model

MODEL_PATH = "outputs/lgbm_model.pkl"


def main() -> None:
    frame = load_california_housing_df()
    model = load_model(MODEL_PATH)
    explainer = build_dice_explainer(model, frame)

    query = frame[FEATURE_COLUMNS].iloc[[0]].copy()
    prediction = model.predict(query)[0]

    target_min = max(prediction + 0.20, 0.05)
    target_max = target_min + 0.30

    cfs = generate_counterfactuals(
        explainer=explainer,
        query_instance=query,
        target_min=target_min,
        target_max=target_max,
        total_cfs=3,
    )

    if cfs.empty:
        raise RuntimeError("No counterfactuals generated. Try a wider target range.")

    original_lat = float(query["Latitude"].iloc[0])
    original_lon = float(query["Longitude"].iloc[0])

    if not (cfs["Latitude"] == original_lat).all():
        raise AssertionError("Latitude changed, but must remain fixed.")
    if not (cfs["Longitude"] == original_lon).all():
        raise AssertionError("Longitude changed, but must remain fixed.")

    print("Smoke test passed.")
    print(f"Generated counterfactuals: {len(cfs)}")


if __name__ == "__main__":
    main()
