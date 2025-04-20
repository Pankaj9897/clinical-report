from flask import Flask, request, render_template
import pickle
import pdfplumber
import pandas as pd
import re

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

expected_features = [
    'Hemoglobin', 'WBC', 'Platelet', 'ESR',
    'Creatinine', 'Urea', 'SGPT_ALT', 'SGPT_AST',
    'TSH', 'Blood_Sugar', 'Cholesterol'
]

# PDF text extraction
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Extract number based on keyword
def extract_number(text, keyword):
    pattern = rf"{keyword}.*?([\d.]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0

# Extract features from text
def extract_features_from_text(text):
    return {
        'Hemoglobin': extract_number(text, 'Hemoglobin'),
        'WBC': extract_number(text, 'WBC'),
        'Platelet': extract_number(text, 'Platelet') / 1000,
        'ESR': extract_number(text, 'ESR'),
        'Creatinine': extract_number(text, 'Creatinine'),
        'Urea': extract_number(text, 'Urea'),
        'SGPT_ALT': extract_number(text, 'SGPT'),
        'SGPT_AST': extract_number(text, 'SGOT'),
        'TSH': extract_number(text, 'TSH'),
        'Blood_Sugar': extract_number(text, 'Blood Sugar'),
        'Cholesterol': extract_number(text, 'Cholesterol')
    }

# Get recommendation
def get_recommendation(disease):
    recommendations = {
        'Anemia': "Take iron-rich food and supplements.",
        'Infection': "Consult doctor for antibiotics.",
        'Thrombocytosis': "Check for clotting issues.",
        'Viral Infection': "Hydrate and rest well.",
        'Normal': "Everything looks good. Stay healthy!"
    }
    return recommendations.get(disease, "Consult a specialist.")

# Main route
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file.save("uploaded_report.pdf")
            text = extract_text_from_pdf("uploaded_report.pdf")
            features = extract_features_from_text(text)

            # Handle useless PDFs (all 0.0 features)
            if all(v == 0.0 for v in features.values()):
                return render_template("result.html",
                                       disease=" Unable to predict",
                                       recommendation="Uploaded PDF does not contain valid blood test values.",
                                       extracted=features,
                                       raw_text=text)

            # Proceed with prediction
            input_data = pd.DataFrame([features])[expected_features]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            recommendation = get_recommendation(prediction)

            return render_template("result.html",
                                   disease=prediction,
                                   recommendation=recommendation,
                                   extracted=features,
                                   raw_text=text)
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)


