# 1. Import necessary libraries
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
import pickle
import numpy as np
import pandas as pd
import pdfplumber
import re
import tempfile
import logging

# --- Configuration ---
# Configure logging to see informational messages
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# --- Load Model and Scaler ---
# Load the pre-trained model and scaler when the application starts.
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model from {MODEL_PATH}: {e}")
    model = None

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load scaler from {SCALER_PATH}: {e}")
    scaler = None

# Define the exact feature names the model expects, in the correct order.
expected_features = [
    'Hemoglobin', 'WBC', 'Platelet', 'ESR',
    'Creatinine', 'Urea', 'SGPT_ALT', 'SGPT_AST',
    'TSH', 'Blood_Sugar', 'Cholesterol'
]

# --- Helper Functions (from your Flask app) ---

def extract_text_from_pdf(path: str) -> str:
    """Extracts all text from a given PDF file."""
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    texts.append(txt)
    except Exception as e:
        logging.exception(f"Error reading PDF: {e}")
    return "\n".join(texts)

def extract_number(text: str, keyword: str, max_chars_after=60) -> float | None:
    """Search for a keyword and extract the nearest numeric value after it."""
    NUMBER_RE = r"([+-]?\d{1,3}(?:[,\d]{0,3})?(?:\.\d+)?)"
    pattern = rf"{re.escape(keyword)}[^\S\r\n]{{0,{max_chars_after}}}.*?{NUMBER_RE}"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        num_text = match.group(1).replace(",", "")
        try:
            return float(num_text)
        except ValueError:
            return None
    return None

def normalize_platelet(value: float | None) -> float | None:
    """Normalizes platelet count (e.g., 250000 -> 250)."""
    if value is None:
        return None
    return value / 1000.0 if value > 2000 else value

def extract_features_from_text(text: str) -> dict:
    """Extracts all required medical features from the text using keywords."""
    keywords = {
        'Hemoglobin': ['Hemoglobin', 'Hb', 'H b'], 'WBC': ['WBC', 'White Blood Cells', 'Leukocytes'],
        'Platelet': ['Platelet', 'Platelets', 'PLT'], 'ESR': ['ESR', 'E.S.R'],
        'Creatinine': ['Creatinine', 'Sr. Creatinine', 'Serum Creatinine'], 'Urea': ['Urea', 'Blood Urea', 'BUN'],
        'SGPT_ALT': ['SGPT', 'ALT'], 'SGPT_AST': ['SGOT', 'AST'], 'TSH': ['TSH', 'Thyroid Stimulating Hormone'],
        'Blood_Sugar': ['Blood Sugar', 'Glucose', 'Fasting Blood Sugar', 'FBS'],
        'Cholesterol': ['Cholesterol', 'Total Cholesterol']
    }
    extracted = {}
    for feat, kws in keywords.items():
        val = None
        for k in kws:
            val = extract_number(text, k)
            if val is not None:
                break
        if feat == 'Platelet':
            val = normalize_platelet(val)
        extracted[feat] = float(val) if val is not None else np.nan
    return extracted

def get_recommendation(disease: str) -> str:
    """Returns a health recommendation based on the predicted disease."""
    recommendations = {
        'Anemia': "Take iron-rich food and supplements. Consult a physician for confirmation.",
        'Infection': "Consult doctor for antibiotics and further tests.",
        'Thrombocytosis': "Check for clotting issues; follow-up with hematologist.",
        'Viral Infection': "Hydrate, rest, and consult if fever persists.",
        'Normal': "Everything looks good. Stay healthy!"
    }
    return recommendations.get(disease, "Consult a specialist for further evaluation.")

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Blood Report Analysis API. Please use the /predict endpoint to upload a PDF."}

@app.post("/predict/")
async def predict_from_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, extracts features, and returns a disease prediction.
    """
    if model is None or scaler is None:
        return {"error": "Model or scaler not available on the server. Please check logs."}
    
    # Use a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        text = extract_text_from_pdf(tmp_path)
        if not text:
            return {"error": "Could not extract any text from the PDF."}

        features = extract_features_from_text(text)

        # Build DataFrame in the expected order for the model
        input_data = pd.DataFrame([features])[expected_features]

        # Handle missing values using a simple imputation (filling with mean)
        if input_data.isnull().values.any():
            # For simplicity, we'll fill with 0, but a more robust method would be to use column means
            # from the training set if available.
            input_data = input_data.fillna(0)

        # Scale the features and make a prediction
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        recommendation = get_recommendation(prediction)

        return {
            "prediction": str(prediction),
            "recommendation": recommendation,
            "extracted_features": {k: (v if not np.isnan(v) else None) for k, v in features.items()},
            "raw_text_summary": text[:1000] + "..." # Return a summary of the text
        }
    except Exception as e:
        logging.exception(f"Prediction error: {e}")
        return {"error": f"An error occurred during processing: {e}"}
    finally:
        # Clean up the temporary file
        os.remove(tmp_path)

# This block allows running the app directly with `python main.py` for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


