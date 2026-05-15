import joblib
import os
import re

# Ambil lokasi folder file ini berada (src/toxic)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Susun path ke file model secara absolut
model_path = os.path.join(BASE_DIR, "..", "..", "models", "toxic", "toxic_model.pkl")

# =========================
# LOAD MODEL
# =========================
model = joblib.load(model_path)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# =========================
# TOXIC KEYWORDS DICTIONARY
# =========================
TOXIC_KEYWORDS = [
    "bangsat",
    "anjing",
    "tolol",
    "goblok",
    "kontol",
    "memek",
    "bajingan",
    "dongo",
    "dongok",
    "sialan",
    "setan",
    "brengsek",
    "monyet",
    "ngentot",
    "ngentod",
    "kentot",
    "bacot",
    "bacod",
    "perek",
    "bego"
]

# =========================
# PREDICT FUNCTION
# =========================
def predict_toxic(message):
    cleaned = clean_text(message)

    # Method 1: Quick keyword matching
    for word in TOXIC_KEYWORDS:
        if word in cleaned:
            return {
                "is_toxic": True,
                "toxic_score": 1.0,
                "method": "keyword"
            }

    # Method 2: ML Model prediction
    try:
        prediction = model.predict([cleaned])[0]
        probabilities = model.predict_proba([cleaned])[0][1]
        toxic_score = float(probabilities[1])

        return {
            "is_toxic": bool(prediction),
            "toxic_score": toxic_score,
            "method": "model"
        }
    except Exception as e:
        import logging
        logging.error(f"Model prediction error: {str(e)}")
        return {
            "is_toxic": False,
            "toxic_score": 0.0,
            "method": "error"
        }


# =========================
# TEST
# =========================
if __name__ == "__main__":
    text = input("Enter message: ")
    result = predict_toxic(text)
    print(result)