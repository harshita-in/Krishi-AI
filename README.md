# Krishi AI

Krishi AI is an agriculture assistant that combines locally trained machine-learning models with a Groq-powered conversational chatbot.

## Features

- Crop recommendation from soil and weather measurements.
- Crop disease detection from a local plant image.
- Crop modal-price prediction.
- Groq-powered natural-language conversations for farmers.
- Browser chatbot with Markdown-formatted responses.
- GPU-enabled PyTorch support for the optional MiniGPT experiments.

## Requirements

- Windows with Python 3.14 or a compatible Python version.
- The project virtual environment at `venv`.
- A CUDA-capable NVIDIA GPU is optional.
- A Groq API key is required for the chatbot.

Install dependencies into the project environment:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configure Groq

Create or edit the root `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is ignored by Git. Never commit API keys or other credentials.

## Run the web chatbot

From the repository root:

```powershell
.\venv\Scripts\python.exe server.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser.

The chatbot can:

- Answer general farming questions through Groq.
- Route crop-recommendation requests to the local recommendation model.
- Route price-prediction requests to the local CatBoost model.
- Accept a plant image and route disease detection to the local CNN model.

For disease detection, attach an image using the paperclip button in the web interface.

## Model training

### Crop recommendation

```powershell
.\venv\Scripts\python.exe crop_recommendation\train\train.py
```

The trainer reads the cached KaggleHub dataset and saves:

```text
crop_recommendation\model\crop_recommendation_model.pkl
```

### Crop disease detection

```powershell
.\venv\Scripts\python.exe crop_disease_detection\train\train.py
```

The trainer reads the cached plant-disease dataset and saves:

```text
crop_disease_detection\model\crop_disease_cnn_model.keras
```

### Crop price prediction

```powershell
.\venv\Scripts\python.exe crop_price_prediction\train\train.py
```

The trainer reads the cached agricultural mandi-price dataset and saves:

```text
crop_price_prediction\model\crop_price_prediction_model.cbm
```

All trainers use datasets from:

```text
C:\Users\shrey\.cache\kagglehub\datasets
```

The required cached datasets are:

- `atharvaingle\crop-recommendation-dataset`
- `vipoooool\new-plant-diseases-dataset`
- `arjunyadav99\indian-agricultural-mandi-prices-20232025`

## Test the models

Crop recommendation:

```powershell
.\venv\Scripts\python.exe crop_recommendation\test\test_model.py
```

Crop price prediction for the sample CSV:

```powershell
.\venv\Scripts\python.exe crop_price_prediction\test\test_csv.py
```

The price CSV test writes predictions to:

```text
crop_price_prediction\test\test_predictions.csv
```

Crop disease detection can be run directly and will prompt for a local image:

```powershell
.\venv\Scripts\python.exe crop_disease_detection\test\test.py
```

## Optional MiniGPT experiment

The character-level MiniGPT files are experimental and separate from the Groq chatbot:

```powershell
cd native_language_nlp\train
..\..\venv\Scripts\python.exe train.py
..\..\venv\Scripts\python.exe generate.py
```

The current production chatbot uses Groq instead of MiniGPT.

## Project structure

```text
server.py                         Root web-server entry point
native_language_nlp/
  groq.py                         Groq routing and local-model integration
  web/
    index.html                    Chatbot page
    styles.css                    Chatbot styling
    app.js                        Browser interaction and Markdown rendering
crop_recommendation/
  model/                          Saved recommendation model
  train/                          Recommendation training script
crop_disease_detection/
  model/                          Saved disease model
  train/                          Disease training script
  test/                           Image prediction script
crop_price_prediction/
  model/                          Saved price model
  train/                          Price training script
  test/                           CSV and single-row prediction scripts
```

## Security

- Keep `.env` local and never commit it.
- Keep the Groq API key on the Python server; do not place it in browser JavaScript.
- Validate uploaded images and user-provided model inputs before production deployment.

