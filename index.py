# 1. Import necessary libraries
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
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

# --- HTML Templates ---

def get_error_page(message: str) -> str:
    """Returns an HTML page for displaying an error."""
    return f"""
    <html>
        <head><title>Error</title></head>
        <body>
            <h1>An Error Occurred</h1>
            <p>{message}</p>
            <a href="/">Try again</a>
        </body>
    </html>
    """

# --- API Endpoints with UI ---

@app.get("/", response_class=HTMLResponse)
async def upload_page():
    """Serves the main page with the file upload form."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Blood Report Analyzer</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background-color: #f4f7f6; }
            .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            input[type="file"] { border: 1px solid #ccc; padding: 10px; border-radius: 4px; }
            input[type="submit"] { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
            input[type="submit"]:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Upload Blood Report PDF</h1>
            <form action="/predict/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".pdf" required>
                <br><br>
                <input type="submit" value="Analyze Report">
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/", response_class=HTMLResponse)
async def predict_and_show_results(file: UploadFile = File(...)):
    """Accepts a PDF, processes it, and displays the results on a new HTML page."""
    if model is None or scaler is None:
        return HTMLResponse(content=get_error_page("Model or scaler not available on the server."), status_code=500)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        text = extract_text_from_pdf(tmp_path)
        if not text:
            return HTMLResponse(content=get_error_page("Could not extract any text from the PDF."), status_code=400)

        features = extract_features_from_text(text)
        input_data = pd.DataFrame([features])[expected_features]

        if input_data.isnull().values.any():
            input_data = input_data.fillna(0)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        recommendation = get_recommendation(prediction)

        # Build the HTML table for extracted features
        features_html = ""
        for key, value in features.items():
            display_value = round(value, 2) if isinstance(value, (int, float)) and not np.isnan(value) else "<strong>Not Found</strong>"
            features_html += f"<tr><td>{key}</td><td>{display_value}</td></tr>"

        # The final HTML page to display results
        result_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Result</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background-color: #f4f7f6; }}
                .container {{ max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #333; }}
                .prediction {{ color: #007bff; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
                th {{ background-color: #007bff; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                a {{ color: #007bff; text-decoration: none; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Result</h1>
                <h2>Prediction: <span class="prediction">{prediction}</span></h2>
                <p><strong>Recommendation:</strong> {recommendation}</p>
                
                <h3>Extracted Values from PDF</h3>
                <table>
                    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                    <tbody>{features_html}</tbody>
                </table>
                <br>
                <a href="/">Upload Another Report</a>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=result_html)
    except Exception as e:
        logging.exception(f"Prediction error: {e}")
        return HTMLResponse(content=get_error_page(f"An error occurred during processing: {e}"), status_code=500)
    finally:
        os.remove(tmp_path)

# This block allows running the app directly with `python main.py` for local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)




