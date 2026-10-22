import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hides ALL logs (Info, Warnings, Errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: Turns off the oneDNN notice

import numpy as np
import tensorflow as tf
import kagglehub

# ==========================================
# 1. Load the Saved CNN Model
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "model", "crop_disease_cnn_model.keras")

if not os.path.exists(model_path):
    print(f"Error: Model not found at '{model_path}'. Please train and save your CNN model first!")
    exit()

print("Loading trained CNN model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!\n")


# ==========================================
# 2. Retrieve Class Names (Alphabetical Order)
# ==========================================
print("Locating dataset to load class names...")
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
train_dir = None
for root, dirs, files in os.walk(path):
    if "train" in dirs:
        train_dir = os.path.join(root, "train")
        break

if not train_dir:
    for root, dirs, files in os.walk(path):
        if os.path.basename(root) == "train":
            train_dir = root
            break

# TensorFlow maps class folders alphabetically by default
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print(f"Successfully loaded {len(class_names)} class labels.\n")


# ==========================================
# 3. Specify Your Test Image Path
# ==========================================
# Put the path to any leaf image on your computer here
test_image_path = r"C:\Users\shrey\Desktop\Krishi-AI\sample_leaf.JPG"

if not os.path.exists(test_image_path):
    print(f"❌ Error: Image file not found at '{test_image_path}'.")
    print("👉 Please save a leaf image locally and update the 'test_image_path' variable in this script.")
    exit()


# ==========================================
# 4. Preprocess the Single Image
# ==========================================
IMG_SIZE = 128
print(f"Processing image: {test_image_path}")

# Load image and resize it to 128x128 pixels (the size used during training)
image = tf.keras.utils.load_img(test_image_path, target_size=(IMG_SIZE, IMG_SIZE))

# Convert image into a numerical array
input_arr = tf.keras.utils.img_to_array(image)

# Expand dimensions to create a batch of 1 (Model expects shape: [1, 128, 128, 3])
input_arr = np.array([input_arr]) 

# Normalize pixel values from [0, 255] down to [0.0, 1.0]
input_arr = input_arr / 255.0


# ==========================================
# 5. Make Prediction
# ==========================================
print("Running prediction...")
predictions = model.predict(input_arr)

# Find the index of the highest probability
predicted_class_index = np.argmax(predictions[0])
predicted_crop_disease = class_names[predicted_class_index]
confidence_score = float(np.max(predictions[0])) * 100


# ==========================================
# 6. Display Results
# ==========================================
print("\n========================================")
print(f" 🌱 Predicted Disease/Status : {predicted_crop_disease}")
print(f" 📊 Confidence Score          : {confidence_score:.2f}%")
print("========================================")