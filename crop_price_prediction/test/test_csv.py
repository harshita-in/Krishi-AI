import os
import sys

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
    script_dir, "..", "model", "crop_price_prediction_model.cbm"
)
input_path = (
    sys.argv[1]
    if len(sys.argv) == 2
    else os.path.join(script_dir, "test_data.csv")
)
output_path = os.path.join(script_dir, "test_predictions.csv")

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"Model not found at '{model_path}'. Run the training script first."
    )

if not os.path.isfile(input_path):
    raise FileNotFoundError(f"Input CSV not found at '{input_path}'.")

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
required_columns = categorical_features + [
    "Price Date",
    "Min_Price",
    "Max_Price",
]

data = pd.read_csv(input_path)
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    raise ValueError(
        f"Input CSV is missing required columns: {missing_columns}. "
        f"Available columns: {list(data.columns)}"
    )

data["Price Date"] = pd.to_datetime(
    data["Price Date"], format="mixed", errors="coerce"
)
if data["Price Date"].isna().any():
    raise ValueError("Input CSV contains invalid values in 'Price Date'.")

data["Year"] = data["Price Date"].dt.year
data["Month"] = data["Price Date"].dt.month
data["Day"] = data["Price Date"].dt.day
data["DayOfWeek"] = data["Price Date"].dt.dayofweek

feature_columns = categorical_features + numerical_features
features = data[feature_columns].copy()
for column in categorical_features:
    features[column] = features[column].astype(str)

model = CatBoostRegressor()
model.load_model(model_path)
data["Predicted_Modal_Price"] = model.predict(features)
data.to_csv(output_path, index=False)

print(f"Loaded model from: {model_path}")
print(f"Processed {len(data)} rows from: {input_path}")
print(f"Predictions saved to: {output_path}")
print("\nPredictions:")
result_columns = ["Commodity", "Price Date", "Modal_Price", "Predicted_Modal_Price"]
available_result_columns = [
    column for column in result_columns if column in data.columns
]
print(data[available_result_columns].to_string(index=False))

if "Modal_Price" in data.columns:
    actual = data["Modal_Price"]
    predicted = data["Predicted_Modal_Price"]
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    non_zero_actual = actual != 0
    percentage_errors = (
        (actual[non_zero_actual] - predicted[non_zero_actual]).abs()
        / actual[non_zero_actual]
    )
    mape = percentage_errors.mean() * 100
    approximate_accuracy = max(0.0, 100.0 - mape)
    within_5_percent = (percentage_errors <= 0.05).mean() * 100
    within_10_percent = (percentage_errors <= 0.10).mean() * 100

    print("\nEvaluation Metrics")
    print("-" * 42)
    print(f"{'Metric':<30} {'Result':>10}")
    print("-" * 42)
    print(f"{'R2 score':<30} {r2:>10.4f}")
    print(f"{'MAPE':<30} {mape:>9.2f}%")
    print(f"{'MAE':<30} {mae:>10.2f}")
    print(f"{'RMSE':<30} {rmse:>10.2f}")
    print(f"{'Within +/-5%':<30} {within_5_percent:>9.2f}%")
    print(f"{'Within +/-10%':<30} {within_10_percent:>9.2f}%")
    print("-" * 42)
    print(f"Final approximate accuracy: {approximate_accuracy:.2f}%")
else:
    print(
        "\nModal_Price is not present in the input CSV; "
        "evaluation metrics were not calculated."
    )
