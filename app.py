from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

app = Flask(__name__)

MODEL_PATH = "model/churn_model.pkl"
SCHEMA_PATH = "model/input_schema.json"

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH, "r") as f:
    schema = json.load(f)

REQUIRED_FIELDS = schema["required_fields"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0)
    out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce").fillna(0)

    out["NumServices"] = 0
    out["NumServices"] += (out["PhoneService"].astype(str) == "Yes").astype(int)
    out["NumServices"] += (out["MultipleLines"].astype(str) == "Yes").astype(int)
    out["NumServices"] += (out["OnlineSecurity"].astype(str) == "Yes").astype(int)
    out["NumServices"] += (out["OnlineBackup"].astype(str) == "Yes").astype(int)
    out["NumServices"] += (out["DeviceProtection"].astype(str) == "Yes").astype(int)
    out["NumServices"] += (out["TechSupport"].astype(str) == "Yes").astype(int)
    out["NumServices"] += (out["StreamingTV"].astype(str) == "Yes").astype(int)
    out["NumServices"] += (out["StreamingMovies"].astype(str) == "Yes").astype(int)

    tenure_safe = out["tenure"].replace(0, 1)
    out["AvgChargesPerMonth"] = out["TotalCharges"] / tenure_safe
    out["IsNewCustomer"] = (out["tenure"] <= 12).astype(int)

    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])

    return out


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Customer Churn API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        if payload is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        missing_fields = [col for col in REQUIRED_FIELDS if col not in payload]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        row = {col: payload[col] for col in REQUIRED_FIELDS}
        input_df = pd.DataFrame([row])
        input_df = add_engineered_features(input_df)

        prediction = model.predict(input_df)[0]
        classes = list(model.named_steps["model"].classes_)
        yes_idx = classes.index("Yes")
        churn_prob = float(model.predict_proba(input_df)[0][yes_idx])

        return jsonify({
            "prediction": str(prediction),
            "churn_probability": round(churn_prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)