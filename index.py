from flask import Flask, request, render_template, jsonify
import pickle
import pdfplumber
import pandas as pd
import re
import tempfile
import os
import numpy as np
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB max upload

logging.basicConfig(level=logging.INFO)

# Load model and scaler safely
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded.")
except Exception as e:
    logging.error("Failed to load model: %s", e)
    model = None

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    logging.info("Scaler loaded.")
except Exception as e:
    logging.error("Failed to load scaler: %s", e)
    scaler = None

expected_features = [
    'Hemoglobin', 'WBC', 'Platelet', 'ESR',
    'Creatinine', 'Urea', 'SGPT_ALT', 'SGPT_AST',
    'TSH', 'Blood_Sugar', 'Cholesterol'
]

# Robust text extraction from PDF
def extract_text_from_pdf(path):
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    texts.append(txt + "\n")
    except Exception as e:
        logging.exception("Error reading PDF: %s", e)
    return "\n".join(texts)

# Robust number extractor: finds the nearest number after keyword
NUMBER_RE = r"([+-]?\d{1,3}(?:[,\d]{0,3})?(?:\.\d+)?)"

def extract_number(text, keyword, max_chars_after=60):
    """
    Search for 'keyword' and extract the nearest numeric token after it.
    Returns float or None.
    """
    # build pattern: keyword ... up to max_chars_after ... capture number
    pattern = rf"{re.escape(keyword)}[^\S\r\n]{{0,{max_chars_after}}}.*?{NUMBER_RE}"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        num_text = match.group(1)
        # normalize comma like "1,234" -> "1234"
        num_text = num_text.replace(",", "")
        try:
            return float(num_text)
        except:
            return None
    # fallback: try any occurrence of keyword then any number after on same line
    line_pattern = rf"^{re.escape(keyword)}.*?{NUMBER_RE}"
    for line in text.splitlines():
        m = re.search(line_pattern, line, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except:
                return None
    return None

def normalize_platelet(value):
    """
    Many reports show Platelet as per µL (e.g., 250000).
    Our model expects platelet in thousands (e.g., 250).
    Heuristic: if value > 2000 -> divide by 1000
    """
    if value is None:
        return None
    if value > 2000:
        return value / 1000.0
    return value

def extract_features_from_text(text):
    # Keywords map: feature -> list of possible keywords in reports
    keywords = {
        'Hemoglobin': ['Hemoglobin', 'Hb', 'H b'],
        'WBC': ['WBC', 'White Blood Cells', 'Leukocytes'],
        'Platelet': ['Platelet', 'Platelets', 'PLT'],
        'ESR': ['ESR', 'E.S.R'],
        'Creatinine': ['Creatinine', 'Sr. Creatinine', 'Serum Creatinine'],
        'Urea': ['Urea', 'Blood Urea', 'BUN'],
        'SGPT_ALT': ['SGPT', 'ALT'],
        'SGPT_AST': ['SGOT', 'AST'],
        'TSH': ['TSH', 'Thyroid Stimulating Hormone'],
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
        # platelet normalization
        if feat == 'Platelet':
            val = normalize_platelet(val)
        extracted[feat] = float(val) if (val is not None) else 0.0

    return extracted

def get_recommendation(disease):
    recommendations = {
        'Anemia': "Take iron-rich food and supplements. Consult a physician for confirmation.",
        'Infection': "Consult doctor for antibiotics and further tests.",
        'Thrombocytosis': "Check for clotting issues; follow-up with hematologist.",
        'Viral Infection': "Hydrate, rest, and consult if fever persists.",
        'Normal': "Everything looks good. Stay healthy!"
    }
    return recommendations.get(disease, "Consult a specialist.")

# HTML upload page route
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("result.html",
                                   disease="Unable to predict",
                                   recommendation="No file uploaded.",
                                   extracted={},
                                   raw_text="")
        filename = secure_filename(file.filename or "report.pdf")
        # use tempfile to avoid collisions and auto-clean
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        try:
            text = extract_text_from_pdf(tmp_path)
            features = extract_features_from_text(text)

            if all(v == 0.0 for v in features.values()):
                return render_template("result.html",
                                       disease="Unable to predict",
                                       recommendation="Uploaded PDF does not contain valid blood test values or format not recognized.",
                                       extracted=features,
                                       raw_text=text)

            # ensure model/scaler loaded
            if model is None or scaler is None:
                return render_template("result.html",
                                       disease="Error",
                                       recommendation="Model or scaler not available on server.",
                                       extracted=features,
                                       raw_text=text)

            # build dataframe in expected order
            input_data = pd.DataFrame([features])[expected_features]
            # handle missing/nan by simple imputation (mean of column) to avoid scaler errors
            input_data = input_data.replace(0.0, np.nan)
            if input_data.isnull().any().any():
                # simple impute with column means (or zeros) - here we use column means (if scaler expects)
                input_data = input_data.fillna(input_data.mean().fillna(0))

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            recommendation = get_recommendation(prediction)

            return render_template("result.html",
                                   disease=prediction,
                                   recommendation=recommendation,
                                   extracted=features,
                                   raw_text=text)
        except Exception as e:
            logging.exception("Prediction error: %s", e)
            return render_template("result.html",
                                   disease="Error",
                                   recommendation=f"An error occurred during processing: {e}",
                                   extracted={},
                                   raw_text="")
        finally:
            # cleanup temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return render_template("upload.html")

# Optional JSON API for programmatic usage
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts multipart form upload with 'file' or raw PDF bytes.
    Returns JSON with prediction and extracted values.
    """
    if 'file' not in request.files:
        return jsonify({"error": "no file provided"}), 400
    file = request.files['file']
    if not file:
        return jsonify({"error": "no file"}), 400
    filename = secure_filename(file.filename or "report.pdf")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        text = extract_text_from_pdf(tmp_path)
        features = extract_features_from_text(text)

        if all(v == 0.0 for v in features.values()):
            return jsonify({
                "disease": None,
                "recommendation": "Uploaded PDF does not contain valid blood test values.",
                "extracted": features,
                "raw_text": text
            }), 400

        if model is None or scaler is None:
            return jsonify({"error": "model/scaler not available"}), 500

        input_data = pd.DataFrame([features])[expected_features]
        input_data = input_data.replace(0.0, np.nan)
        input_data = input_data.fillna(input_data.mean().fillna(0))
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        recommendation = get_recommendation(prediction)

        return jsonify({
            "disease": str(prediction),
            "recommendation": recommendation,
            "extracted": features,
            "raw_text": text
        })
    except Exception as e:
        logging.exception("API predict error: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
app.handler = app
