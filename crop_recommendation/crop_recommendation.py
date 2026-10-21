import os
import joblib
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Download dataset using kagglehub
path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")
print("Path to dataset files:", path)

# 2. Locate the CSV file dynamically (passing 'path' to os.walk)
csv_file_path = None
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.csv'):
            csv_file_path = os.path.join(root, file)
            break

print(f"Dataset file found at: {csv_file_path}")

# 3. Load the dataset using pandas
df = pd.read_csv(csv_file_path)

print("\nFirst 5 rows of the dataset:")
print(df.head())

# 4. Define Features (X) and Target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 5. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and Train the Random Forest Classifier
print("\nTraining the Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Training completed!")

# 7. Evaluate the Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Make a Prediction with New Input Data
new_sample = pd.DataFrame(
    [[90, 42, 43, 20.87, 82.0, 6.5, 202.93]], 
    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
)

predicted_crop = rf_model.predict(new_sample)
print(f"\nRecommended Crop for the given soil/weather conditions: **{predicted_crop[0].upper()}**")

script_directory = os.path.dirname(os.path.abspath(__file__))

# Combine directory and filename
model_filename = os.path.join(script_directory, "crop_recommendation_model.pkl")
joblib.dump(rf_model, model_filename)
print(f"Model successfully saved at: {model_filename}")