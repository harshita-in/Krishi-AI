import os
import sys

import pandas as pd
from catboost import CatBoostRegressor


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    script_dir, "..", "model", "crop_price_prediction_model.cbm"
)

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"Model not found at '{model_path}'. Run the training script first."
    )

categorical_features = [
    "STATE",
    "District Name",
    "Market Name",
    "Commodity",
    "Variety",
    "Grade",
]
numerical_features = [
    "Min_Price",
    "Max_Price",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
]


def build_input(values):
    """Build a prediction row using the same features as the training script."""
    input_data = {
        "STATE": values[0],
        "District Name": values[1],
        "Market Name": values[2],
        "Commodity": values[3],
        "Variety": values[4],
        "Grade": values[5],
        "Min_Price": float(values[6]),
        "Max_Price": float(values[7]),
        "Year": int(values[8]),
        "Month": int(values[9]),
        "Day": int(values[10]),
        "DayOfWeek": int(values[11]),
    }
    return pd.DataFrame([input_data], columns=categorical_features + numerical_features)


default_values = [
    "Maharashtra",
    "nashik",
    "Lasalgaon(Niphad)",
    "Wheat",
    "Maharashtra 2189",
    "FAQ",
    2172,
    2399,
    2023,
    6,
    6,
    1,
]

if len(sys.argv) == 13:
    input_values = sys.argv[1:]
elif len(sys.argv) == 1:
    input_values = default_values
else:
    raise ValueError(
        "Provide either no arguments or exactly 12 values: "
        "STATE District Market Commodity Variety Grade "
        "Min_Price Max_Price Year Month Day DayOfWeek"
    )

model = CatBoostRegressor()
model.load_model(model_path)
prediction = float(model.predict(build_input(input_values))[0])

print(f"Loaded model from: {model_path}")
print(f"Predicted modal price: {prediction:.2f}")