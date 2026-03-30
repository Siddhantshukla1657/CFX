from __future__ import annotations

from src.data_loader import train_test_data
from src.model import evaluate_model, save_model, train_lightgbm_model

MODEL_OUTPUT_PATH = "outputs/lgbm_model.pkl"
DATA_OUTPUT_PATH = "outputs/california_housing.csv"


def main() -> None:
    x_train, x_test, y_train, y_test, frame = train_test_data()

    model = train_lightgbm_model(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    save_model(model, MODEL_OUTPUT_PATH)
    frame.to_csv(DATA_OUTPUT_PATH, index=False)

    print("Training complete.")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    print(f"Dataset snapshot saved to {DATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
