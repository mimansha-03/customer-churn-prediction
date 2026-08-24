"""
train_model.py
---------------
Loads data/telco_churn.csv, builds a preprocessing + ML pipeline, trains
a churn classifier, evaluates it, and saves everything the Flask app
needs to serve live predictions.

Run:
    python train_model.py
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "../data/telco_churn.csv"
TRAIN_PATH = "../data/telco_churn_train.csv"
TEST_PATH = "../data/telco_churn_test.csv"
MODEL_PATH = "../model/churn_model.pkl"
METRICS_PATH = "../model/metrics.json"
DASHBOARD_DATA_PATH = "../model/dashboard_data.json"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).fillna(df["SeniorCitizen"])
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    return metrics


def main():
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    X_train, y_train = train_df[ALL_FEATURES], train_df["Churn"]
    X_test, y_test = test_df[ALL_FEATURES], test_df["Churn"]

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        results[name] = evaluate(name, pipe, X_test, y_test)
        fitted[name] = pipe

    # pick best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_pipeline = fitted[best_name]
    print(f"\n>>> Selected best model: {best_name}")

    joblib.dump(best_pipeline, MODEL_PATH)

    # feature importances (Random Forest) for the dashboard, if available
    importances = None
    model_step = best_pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        ohe = best_pipeline.named_steps["preprocess"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = NUMERIC_FEATURES + cat_names
        importances = sorted(
            zip(feature_names, model_step.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True,
        )[:12]

    metrics_out = {
        "best_model": best_name,
        "all_results": results,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "top_features": importances,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2, default=float)

    # ---- Precompute dashboard aggregates from the full dataset ----
    raw = pd.read_csv(DATA_PATH)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    raw = raw.dropna(subset=["TotalCharges"])

    def churn_rate_by(col):
        g = raw.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean())
        return {str(k): round(float(v) * 100, 1) for k, v in g.items()}

    tenure_bins = pd.cut(
        raw["tenure"], bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
    )
    tenure_churn = raw.groupby(tenure_bins, observed=True)["Churn"].apply(
        lambda s: (s == "Yes").mean() * 100
    )

    dashboard_data = {
        "total_customers": int(len(raw)),
        "overall_churn_rate": round(float((raw["Churn"] == "Yes").mean()) * 100, 1),
        "avg_monthly_charges": round(float(raw["MonthlyCharges"].mean()), 2),
        "avg_tenure": round(float(raw["tenure"].mean()), 1),
        "churn_by_contract": churn_rate_by("Contract"),
        "churn_by_internet": churn_rate_by("InternetService"),
        "churn_by_payment": churn_rate_by("PaymentMethod"),
        "churn_by_senior": churn_rate_by("SeniorCitizen"),
        "churn_by_tenure_bucket": {str(k): round(float(v), 1) for k, v in tenure_churn.items()},
        "churn_distribution": {
            str(k): int(v) for k, v in raw["Churn"].value_counts().items()
        },
    }
    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved dashboard data -> {DASHBOARD_DATA_PATH}")


if __name__ == "__main__":
    main()
