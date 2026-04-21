from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # FIX: added CORS so browser fetch() doesn't get blocked by preflight

epsilon = 1e-6

# ============================================================
# FEATURE ENGINEERING — must match exactly what was used in training
# ============================================================
def add_features(df, alpha=0.02, T0=25.0):
    R   = df["sensor1"].copy()
    C   = df["sensor2"].copy()
    T   = df["temperature"].copy()
    T_K = T + 273.15

    comp_factor          = 1.0 + alpha * (T - T0)
    df["R_comp"]         = R / (comp_factor + epsilon)
    df["G_comp"]         = 1.0 / (df["R_comp"] + epsilon)
    df["conductance"]    = 1.0 / (R + epsilon)
    df["C_raw"]          = C
    df["C_comp"]         = C / T_K
    df["G_times_C"]      = df["conductance"] * C
    df["G_comp_times_C"] = df["G_comp"] * C
    df["C_over_R_comp"]  = C / (df["R_comp"] + epsilon)
    df["log_G_comp"]     = np.log(df["G_comp"] + epsilon)
    df["log_C"]          = np.log(C + epsilon)
    df["log_G_raw"]      = np.log(df["conductance"] + epsilon)
    df["G_comp_sq"]      = df["G_comp"] ** 2
    df["C_sq"]           = C ** 2
    df["temp_delta"]     = T - T0
    df["temp_sq"]        = (T - T0) ** 2
    df["ratio"]          = R / (C + 1)
    df["diff"]           = R - C
    df["temp_comp1"]     = R / T_K
    df["temp_comp2"]     = C / T_K
    df["s1_log"]         = np.log(R + 1)
    df["s2_log"]         = np.log(C + 1)
    df["cap_res_ratio"]  = C / (R + epsilon)
    return df


# ============================================================
# LOAD MODEL — handles both old format (raw model) and
#              new format (dict with "model" and "features" keys)
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ExtraTrees__physics_full.pkl")

print(f"[BOOT] Loading model from: {MODEL_PATH}")
try:
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "model" in payload:
        model          = payload["model"]
        model_features = payload.get("features", None)
        print(f"[BOOT] ✅ New-format pkl detected.")
        print(f"[BOOT]    Feature list: {model_features}")
    else:
        model          = payload
        model_features = None
        print(f"[BOOT] ✅ Old-format pkl detected (raw estimator).")

except FileNotFoundError:
    print("[BOOT] ❌ ERROR: ExtraTrees__physics_full.pkl not found! Place it next to app.py.")
    model = None
    model_features = None
except Exception as e:
    print(f"[BOOT] ❌ ERROR loading model: {e}")
    model = None
    model_features = None


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def home():
    print("[GET /] Serving index.html")
    return render_template("index.html")


@app.route("/predict", methods=["POST", "OPTIONS"])  # FIX: explicit OPTIONS for CORS preflight
def predict():
    # FIX: handle OPTIONS preflight request
    if request.method == "OPTIONS":
        return jsonify({}), 200

    print("[POST /predict] Request received")

    if model is None:
        return jsonify({"error": "Model not loaded. Check ExtraTrees__physics_full.pkl."}), 500

    try:
        data = request.get_json(force=True)
        print(f"[POST /predict] Payload: {data}")

        # FIX: validate payload is not None before accessing keys
        if not data:
            return jsonify({"error": "Empty or invalid JSON payload."}), 400

        R          = float(data["R"])
        C          = float(data["C"])
        T          = float(data["Temp"])
        min_thresh = float(data.get("min_threshold", 30))
        max_thresh = float(data.get("max_threshold", 70))

        print(f"[POST /predict] Raw inputs → R={R}, C={C}, Temp={T}")

        # ── New-format model: needs feature engineering ──────────
        if model_features is not None:
            df_input = pd.DataFrame([{
                "sensor1":     R,
                "sensor2":     C,
                "temperature": T,
            }])
            df_input  = add_features(df_input)

            missing = [f for f in model_features if f not in df_input.columns]
            if missing:
                msg = f"Feature engineering missing columns: {missing}"
                print(f"[POST /predict] ❌ {msg}")
                return jsonify({"error": msg}), 500

            X        = df_input[model_features]
            moisture = round(float(model.predict(X)[0]), 2)

        # ── Old-format model: expects raw [R, C, T] ──────────────
        else:
            moisture = round(float(model.predict([[R, C, T]])[0]), 2)

        print(f"[POST /predict] Predicted moisture: {moisture}%")

        if moisture < min_thresh:
            status = "water"
        elif moisture > max_thresh:
            status = "stop"
        else:
            status = "ok"

        print(f"[POST /predict] Status: {status}  (min={min_thresh}, max={max_thresh})")
        return jsonify({"moisture": moisture, "status": status})

    except KeyError as e:
        msg = f"Missing field in request: {e}"
        print(f"[POST /predict] ❌ {msg}")
        return jsonify({"error": msg}), 400

    except ValueError as e:
        msg = f"Invalid numeric value: {e}"
        print(f"[POST /predict] ❌ {msg}")
        return jsonify({"error": msg}), 400

    except Exception as e:
        import traceback
        print(f"[POST /predict] ❌ Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
