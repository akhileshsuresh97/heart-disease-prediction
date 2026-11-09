"""Flask web app for the heart disease prediction models.

Run locally:   python app.py
Production:    gunicorn app:app
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
heart_model = joblib.load(ROOT / "models" / "heart_model.joblib")
secondary_model = joblib.load(ROOT / "models" / "secondary_model.joblib")

# name: (label, min, max, is_integer)
NUMERIC_FIELDS = {
    "age": ("Age", 18, 100, True),
    "trestbps": ("Resting blood pressure", 80, 220, True),
    "chol": ("Serum cholesterol", 100, 600, True),
    "thalach": ("Maximum heart rate", 60, 220, True),
    "oldpeak": ("ST depression", 0, 7, False),
    "ca": ("Major vessels coloured", 0, 3, True),
}
CHOICE_FIELDS = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
SECONDARY_LABELS = {
    0: None,
    1: "Hypertension risk profile",
    2: "Diabetes risk profile",
}

app = Flask(__name__)


def parse_form(form) -> tuple[dict, list[str]]:
    values, errors = {}, []
    for name, (label, low, high, is_int) in NUMERIC_FIELDS.items():
        raw = form.get(name, "").strip()
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"{label}: enter a number.")
            continue
        if not low <= value <= high:
            errors.append(f"{label}: enter a value from {low} to {high}.")
            continue
        values[name] = int(value) if is_int else value
    for name, allowed in CHOICE_FIELDS.items():
        try:
            value = int(form.get(name, ""))
        except ValueError:
            value = None
        if value not in allowed:
            errors.append(f"Choose an option for {name}.")
            continue
        values[name] = value
    return values, errors


@app.route("/", methods=["GET", "POST"])
def home():
    result, errors, values = None, [], {}
    if request.method == "POST":
        values, errors = parse_form(request.form)
        if not errors:
            X = pd.DataFrame([values])[FEATURE_ORDER]
            probability = float(heart_model.predict_proba(X)[0, 1])
            result = {"has_disease": probability >= 0.5, "probability": probability,
                      "secondary": None}
            if not result["has_disease"]:
                result["secondary"] = SECONDARY_LABELS[int(secondary_model.predict(X)[0])]
    return render_template("index.html", result=result, errors=errors, values=values)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
