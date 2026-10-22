# Krishi AI Agent Instructions

## Project overview

Krishi AI is a Windows-based agriculture project with three local models and a Groq-backed web chatbot:

- Crop recommendation: scikit-learn Random Forest.
- Crop disease detection: TensorFlow CNN.
- Crop price prediction: CatBoost regressor.
- Conversational interface: Groq API with a Python standard-library web server.

## Repository conventions

- Use Windows paths in commands and documentation.
- Use the existing `venv` environment:
  ```powershell
  .\venv\Scripts\python.exe
  ```
- Do not install packages globally.
- Keep API keys and credentials in the root `.env` file.
- Never commit `.env`, model secrets, downloaded datasets, or credentials.
- Preserve the existing model file locations under each project’s `model` directory.
- Keep frontend assets under `native_language_nlp\web`.
- Keep the web-server entry point at the repository root as `server.py`.

## Dataset conventions

Training scripts must use the local KaggleHub cache:

```text
C:\Users\shrey\.cache\kagglehub\datasets
```

Do not reintroduce automatic dataset downloads when changing the trainers. Fail with a clear error when a required cached dataset is missing.

## Chatbot conventions

- Keep the Groq API key server-side.
- Load configuration from `.env`.
- Route crop recommendation, price prediction, and disease detection to the local models.
- Do not fabricate missing model inputs; ask the user for the required values.
- Keep user messages as plain text in the browser.
- Escape assistant content before rendering Markdown or HTML.
- Do not claim that a local model ran unless it actually returned a result.

## Code changes

- Inspect related scripts and model paths before editing.
- Prefer small, focused changes.
- Reuse existing feature names and preprocessing exactly during inference.
- Surface missing files, invalid inputs, and dependency errors explicitly.
- Avoid broad exception handlers that hide failures.
- Do not modify unrelated working-tree changes.
- Do not commit changes unless explicitly requested.

## Validation

For Python changes, run focused syntax checks:

```powershell
.\venv\Scripts\python.exe -m py_compile path\to\changed_file.py
```

For JavaScript changes, run:

```powershell
node --check path\to\changed_file.js
```

For chatbot changes, verify that:

1. `server.py` starts from the repository root.
2. `http://127.0.0.1:8000` serves the web client.
3. `/api/chat` returns a clear error when `GROQ_API_KEY` is missing.
4. Local model files are found at their documented paths.

Do not run full model training unless it is necessary to validate the requested change.

