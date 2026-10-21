import os
import joblib
import pandas as pd

# 1. Load the trained model using relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(script_dir, "models", "crop_recommendation_model.pkl")

if not os.path.exists(model_filename):
    print(f"Error: Model not found at '{model_filename}'. Please train and save your model first!")
    exit()

print(f"Loading model from: {model_filename}")
loaded_model = joblib.load(model_filename)
print("Model loaded successfully!\n")

# 2. Load the CSV file containing the test data
# (Ensure your CSV file is in the same directory, or update the filename/path below)
csv_file_path = os.path.join(script_dir, "Crop_recommendation.csv")

if not os.path.exists(csv_file_path):
    print(f"Error: CSV file not found at '{csv_file_path}'. Please place your CSV file in the project folder.")
    exit()

df = pd.read_csv(csv_file_path)

# 3. Define the feature columns used during training
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Check if feature columns exist in the CSV
for col in feature_cols:
    if col not in df.columns:
        print(f"Error: Required column '{col}' is missing from the CSV file.")
        exit()

print(f"Starting row-wise predictions for {len(df)} rows...\n")
print(f"{'Row':<6} | {'Actual Crop':<15} | {'Predicted Crop':<15} | {'Status'}")
print("-" * 55)

correct_count = 0
has_labels = 'label' in df.columns

# 4. Iterate through the CSV row-by-row
for index, row in df.iterrows():
    # Extract features for the current single row as a DataFrame
    single_row_features = pd.DataFrame([row[feature_cols]], columns=feature_cols)
    
    # Predict the crop for this row
    predicted_crop = loaded_model.predict(single_row_features)[0]
    
    if has_labels:
        actual_crop = str(row['label'])
        if predicted_crop.lower() == actual_crop.lower():
            status = "✅ Match"
            correct_count += 1
        else:
            status = "❌ Mismatch"
    else:
        actual_crop = "N/A"
        status = "Predicted"

    # Print the row-wise result
    print(f"{index:<6} | {actual_crop:<15} | {predicted_crop:<15} | {status}")

# 5. Display Summary Statistics if labels were present
if has_labels:
    accuracy = (correct_count / len(df)) * 100
    print("-" * 55)
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Total Rows Processed : {len(df)}")
    print(f"Correct Predictions  : {correct_count}")
    print(f"Incorrect Predictions: {len(df) - correct_count}")
    print(f"Overall Row Accuracy : {accuracy:.2f}%")