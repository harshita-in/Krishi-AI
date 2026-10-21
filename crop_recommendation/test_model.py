import os
import joblib
import pandas as pd

# 1. Locate and load the saved model using relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(script_dir, "crop_recommendation_model.pkl")

if not os.path.exists(model_filename):
    print(f"Error: Model not found at '{model_filename}'. Please run your training script first!")
    exit()

print(f"Loading model from: {model_filename}")
loaded_model = joblib.load(model_filename)
print("Model loaded successfully!\n")

# ==========================================
# OPTION 1: Manual Input for a Single Prediction
# ==========================================
print("=== OPTION 1: Testing with Manual Values ===")

# Define feature values: [N, P, K, temperature, humidity, ph, rainfall]
# You can change these numbers to test different soil and weather conditions
sample_data = [[85, 58, 41, 21.77, 80.31, 7.03, 226.65]]

# Convert into a DataFrame with exact column names used during training
manual_input = pd.DataFrame(
    sample_data, 
    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
)

predicted_crop = loaded_model.predict(manual_input)
print(f"Input Soil/Weather Values: {sample_data[0]}")
print(f"Recommended Crop: **{predicted_crop[0].upper()}**\n")


# ==========================================
# OPTION 2: Testing with a Row from a CSV File
# ==========================================
print("=== OPTION 2: Testing with a CSV File Row ===")

# Replace this with the path to any CSV file containing test data
# (For example, your original dataset or a separate test split CSV)
csv_file_path = os.path.join(script_dir, "Crop_recommendation.csv") # Update filename if needed

if os.path.exists(csv_file_path):
    # Load the CSV
    test_df = pd.read_csv(csv_file_path)
    
    # Pick a specific row (e.g., row index 10)
    row_index = 10
    single_row_features = test_df.iloc[[row_index]][['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    actual_crop = test_df.iloc[row_index]['label']
    
    # Make prediction
    csv_prediction = loaded_model.predict(single_row_features)
    
    print(f"Loaded row index {row_index} from CSV.")
    print(f"Actual Crop in CSV: {actual_crop.upper()}")
    print(f"Model Prediction:   {csv_prediction[0].upper()}")
else:
    print(f"CSV file not found at '{csv_file_path}'. (Skipping Option 2 test)")