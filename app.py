from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

epsilon = 1e-6

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def add_features(df, alpha=0.02, T0=25.0):
    R = df["sensor1"].copy()
    C = df["sensor2"].copy()
    T = df["temperature"].copy()
    T_K = T + 273.15

    comp_factor = 1.0 + alpha * (T - T0)
    df["R_comp"] = R / (comp_factor + epsilon)
    df["G_comp"] = 1.0 / (df["R_comp"] + epsilon)
    df["conductance"] = 1.0 / (R + epsilon)
    df["C_raw"] = C
    df["C_comp"] = C / T_K
    df["G_times_C"] = df["conductance"] * C
    df["G_comp_times_C"] = df["G_comp"] * C
    df["C_over_R_comp"] = C / (df["R_comp"] + epsilon)
    df["log_G_comp"] = np.log(df["G_comp"] + epsilon)
    df["log_C"] = np.log(C + epsilon)
    df["G_comp_sq"] = df["G_comp"] ** 2
    df["temp_delta"] = T - T0
    df["temp_sq"] = (T - T0) ** 2

    return df


# ============================================================
# LOAD MODEL
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ExtraTrees__physics_full.pkl")

print(f"[BOOT] Loading model from: {MODEL_PATH}")

try:
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        model_features = payload.get("features", None)
        print("[BOOT] ✅ New-format pkl detected.")
    else:
        model = payload
        model_features = None
        print("[BOOT] ✅ Old-format pkl detected.")

except Exception as e:
    print(f"[BOOT] ❌ Model loading failed: {e}")
    model = None
    model_features = None


# ============================================================
# ROUTES
# ============================================================

# 🔥 FIXED ROOT ROUTE
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"[GET /] Template error: {e}")
        return "Backend is running 🚀"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():

    if request.method == "OPTIONS":
        return jsonify({}), 200

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        R = float(data["R"])
        C = float(data["C"])
        T = float(data["Temp"])
        min_thresh = float(data.get("min_threshold", 30))
        max_thresh = float(data.get("max_threshold", 70))

        # FEATURE ENGINEERING
        if model_features is not None:
            df_input = pd.DataFrame([{
                "sensor1": R,
                "sensor2": C,
                "temperature": T
            }])

            df_input = add_features(df_input)
            X = df_input[model_features]
            moisture = round(float(model.predict(X)[0]), 2)

        else:
            moisture = round(float(model.predict([[R, C, T]])[0]), 2)

        # STATUS LOGIC
        if moisture < min_thresh:
            status = "water"
        elif moisture > max_thresh:
            status = "stop"
        else:
            status = "ok"

        return jsonify({
            "moisture": moisture,
            "status": status
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 400


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
