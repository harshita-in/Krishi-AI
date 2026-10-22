import os
import joblib
import pandas as pd

# 1. Locate and load the saved model using relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_filename = os.path.join(script_dir, "..", "model", "crop_recommendation_model.pkl")

if not os.path.exists(model_filename):
    print(f"Error: Model not found at '{model_filename}'. Please run your training script first!")
    exit()

print(f"Loading model from: {model_filename}")
loaded_model = joblib.load(model_filename)
print("Model loaded successfully!\n")

# ==========================================
# OPTION 1: Manual Input for a Single Prediction
# ==========================================

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