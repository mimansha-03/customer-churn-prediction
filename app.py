"""
Customer Churn Prediction — Flask backend.

Serves:
  GET  /                 -> prediction UI
  GET  /dashboard         -> churn analytics dashboard
  POST /api/predict       -> run the trained model on a customer's details
  GET  /api/dashboard-data -> aggregated churn statistics (for charts)
  GET  /api/model-info    -> model metadata / evaluation metrics
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
DASHBOARD_DATA_PATH = os.path.join(BASE_DIR, "model", "dashboard_data.json")

app = Flask(__name__)

# ---- Load model + metadata once at startup ----
model = joblib.load(MODEL_PATH)

with open(METRICS_PATH) as f:
    METRICS = json.load(f)

with open(DASHBOARD_DATA_PATH) as f:
    DASHBOARD_DATA = json.load(f)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REQUIRED_FIELDS = set(ALL_FEATURES)


def risk_level(probability):
    if probability >= 0.66:
        return "High"
    if probability >= 0.33:
        return "Medium"
    return "Low"


def top_reasons(payload):
    """Lightweight, rule-based explanation of the main churn drivers
    for this specific customer, based on well-established patterns in
    the trained data (contract type, tenure, internet service, charges).
    """
    reasons = []
    if payload["Contract"] == "Month-to-month":
        reasons.append("Month-to-month contract (no long-term commitment)")
    if payload["InternetService"] == "Fiber optic":
        reasons.append("Fiber optic service (historically higher churn segment)")
    if payload["PaymentMethod"] == "Electronic check":
        reasons.append("Pays via electronic check (higher-churn payment method)")
    if float(payload["tenure"]) <= 12:
        reasons.append("Low tenure — still in the early, highest-risk period")
    if payload["OnlineSecurity"] == "No" and payload["InternetService"] != "No":
        reasons.append("No online security add-on")
    if payload["TechSupport"] == "No" and payload["InternetService"] != "No":
        reasons.append("No tech support add-on")
    if float(payload["MonthlyCharges"]) >= 80:
        reasons.append("High monthly charges")
    if payload["PaperlessBilling"] == "Yes" and payload["Contract"] == "Month-to-month":
        reasons.append("Paperless billing combined with a flexible contract")
    if not reasons:
        reasons.append("No major single risk factor — profile looks stable")
    return reasons[:4]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/model-info")
def model_info():
    return jsonify(METRICS)


@app.route("/api/dashboard-data")
def dashboard_data():
    return jsonify(DASHBOARD_DATA)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    missing = REQUIRED_FIELDS - set(payload.keys())
    if missing:
        return jsonify({"error": f"Missing fields: {sorted(missing)}"}), 400

    try:
        row = {}
        for f in NUMERIC_FEATURES:
            row[f] = float(payload[f])
        for f in CATEGORICAL_FEATURES:
            row[f] = str(payload[f])
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X = pd.DataFrame([row])[ALL_FEATURES]

    proba = float(model.predict_proba(X)[0][1])
    prediction = "Churn" if proba >= 0.5 else "No Churn"

    response = {
        "prediction": prediction,
        "churn_probability": round(proba * 100, 1),
        "retain_probability": round((1 - proba) * 100, 1),
        "risk_level": risk_level(proba),
        "key_factors": top_reasons(row),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
