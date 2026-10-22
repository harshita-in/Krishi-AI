import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hides ALL logs (Info, Warnings, Errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: Turns off the oneDNN notice

import numpy as np
import tensorflow as tf
import kagglehub
import cv2

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
# 3. Preprocess and Predict a Camera Frame
# ==========================================
IMG_SIZE = 128
def predict_frame(frame):
    """Resize and normalize a BGR camera frame before classifying it."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    input_arr = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

    predictions = model.predict(input_arr, verbose=0)
    predicted_class_index = int(np.argmax(predictions[0]))
    predicted_disease = class_names[predicted_class_index]
    confidence_score = float(np.max(predictions[0])) * 100
    return predicted_disease, confidence_score


# ==========================================
# 4. Capture and Process Images in Real Time
# ==========================================
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open the camera. Check that it is connected and available.")
    raise SystemExit(1)

print("Camera started.")
print("Press SPACE to classify the current image; press Q to quit.")
result_text = "Press SPACE to classify"

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read a frame from the camera.")
            break

        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            result_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Crop Disease Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            predicted_disease, confidence_score = predict_frame(frame)
            result_text = f"{predicted_disease} ({confidence_score:.2f}%)"
            print(f"Predicted Disease/Status: {predicted_disease}")
            print(f"Confidence Score: {confidence_score:.2f}%")
finally:
    camera.release()
    cv2.destroyAllWindows()