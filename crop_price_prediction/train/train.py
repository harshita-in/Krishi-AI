import pandas as pd
import numpy as np
import os
import sys
import kagglehub
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Use an optional command-line argument for a local dataset directory.
# Otherwise, download the dataset configured by the project.
script_dir = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]
else:
    print("Downloading agricultural mandi prices dataset...")
    dataset_path = kagglehub.dataset_download(
        "arjunyadav99/indian-agricultural-mandi-prices-20232025"
    )
    print(f"Dataset downloaded to: {dataset_path}")

csv_files = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(dataset_path)
    for file in files
    if file.lower().endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError(
        f"No CSV dataset found in '{dataset_path}'. "
        "Place the dataset there or pass its directory as the first argument."
    )

df = pd.read_csv(csv_files[0])
print(f"Using dataset: {csv_files[0]}")

required_columns = [
    'Price Date',
    'STATE',
    'District Name',
    'Market Name',
    'Commodity',
    'Variety',
    'Grade',
    'Min_Price',
    'Max_Price',
    'Modal_Price',
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}. "
        f"Available columns: {list(df.columns)}"
    )

# Quick preprocessing: Handle dates and missing values
df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')
df['Year'] = df['Price Date'].dt.year
df['Month'] = df['Price Date'].dt.month
df['Day'] = df['Price Date'].dt.day
df['DayOfWeek'] = df['Price Date'].dt.dayofweek

# Define Features and Target
target = 'Modal_Price'
categorical_features = ['STATE', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade']
numerical_features = ['Min_Price', 'Max_Price', 'Year', 'Month', 'Day', 'DayOfWeek']

# Drop rows with missing targets or essential features
df = df.dropna(subset=[target] + categorical_features)

X = df[categorical_features + numerical_features]
y = df[target]

# Ensure categorical columns are strings
for col in categorical_features:
    X[col] = X[col].astype(str)

# Train/Test Split (Time-based split is ideal, but random split works for a baseline)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost Regressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    cat_features=categorical_features,
    eval_metric='MAE',
    random_seed=42,
    early_stopping_rounds=50
)

# Train the model
print("Training CatBoost model...")
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)

# Evaluate
predictions = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R2 Score: {r2_score(y_test, predictions):.4f}")

# Save the trained model in the project model directory.
model_dir = os.path.join(script_dir, "..", "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "crop_price_prediction_model.cbm")
model.save_model(model_path)
print(f"Model successfully saved at: {model_path}")