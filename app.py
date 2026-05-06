from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "../models/model_rf.pkl"

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# =========================
# RULE-BASED RESOURCE ESTIMATION
# =========================

def estimate_resources(complexity):
    if complexity == "low":
        return 2, 1
    elif complexity == "medium":
        return 3, 2
    elif complexity == "high":
        return 5, 4
    else:
        return 3, 2  # default

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return "ML API is running..."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # =========================
        # VALIDASI INPUT
        # =========================
        required_fields = [
            'product_type', 'material_type',
            'length_m', 'width_m', 'height_m',
            'complexity_level', 'quantity',
            'finishing_type', 'installation',
            'location_type'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400

        # =========================
        # HITUNG FIELD TURUNAN
        # =========================
        volume = data['length_m'] * data['width_m'] * data['height_m']

        # =========================
        # AUTO GENERATE
        # =========================
        num_workers, production_days = estimate_resources(data['complexity_level'])

        # =========================
        # BENTUK DATAFRAME
        # =========================
        input_data = {
            'product_type': data['product_type'],
            'material_type': data['material_type'],
            'length_m': data['length_m'],
            'width_m': data['width_m'],
            'height_m': data['height_m'],
            'volume_m3': volume,
            'complexity_level': data['complexity_level'],
            'quantity': data['quantity'],
            'num_workers': num_workers,
            'production_days': production_days,
            'finishing_type': data['finishing_type'],
            'installation': data['installation'],
            'location_type': data['location_type']
        }

        df = pd.DataFrame([input_data])

        # =========================
        # PREDIKSI
        # =========================
        prediction = model.predict(df)[0]

        return jsonify({
            "predicted_cost": int(prediction),
            "estimated_workers": num_workers,
            "estimated_days": production_days
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)