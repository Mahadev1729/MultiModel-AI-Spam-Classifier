from flask import Flask, request, jsonify, render_template
import joblib
import re
import librosa
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ===========================
# LOAD MODELS
# ===========================

# SMS
sms_model = joblib.load("svm_sms_model.pkl")
sms_tfidf = joblib.load("tfidf_vectorizer.pkl")

# Email
email_model = joblib.load("svm_email_model1.pkl")
email_tfidf = joblib.load("tfidf_email_vectorizer1.pkl")

# WhatsApp
whatsapp_model = joblib.load("svm_whatsapp_model.pkl")
whatsapp_tfidf = joblib.load("tfidf_whatsapp_vectorizer.pkl")

# Voice
voice_model = joblib.load("voice.pkl")
voice_scaler = joblib.load("scaler.pkl")
voice_encoder = joblib.load("encoder.pkl")


# ===========================
# TEXT CLEANING (IMPORTANT)
# ===========================

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)          # remove emails
    text = re.sub(r"[^a-z\s]", " ", text)         # remove numbers/symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================
# FRONTEND ROUTES
# ===========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sms")
def sms_page():
    return render_template("sms.html")

@app.route("/email")
def email_page():
    return render_template("email.html")

@app.route("/whatsapp")
def whatsapp_page():
    return render_template("whatsapp.html")

@app.route("/voice")
def voice_page():
    return render_template("voice.html")


# ===========================
# AUDIO FEATURE EXTRACTION
# ===========================

def extract_voice_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    features = {
        "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
    }

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc{i+1}"] = np.mean(mfccs[i])

    return pd.DataFrame([features])


# ===========================
# API ROUTES
# ===========================

# ----- SMS -----
@app.route("/predict_sms", methods=["POST"])
def predict_sms():
    data = request.get_json()
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    message = clean_text(message)
    X = sms_tfidf.transform([message])
    pred = sms_model.predict(X)[0]

    return jsonify({"prediction": "SPAM" if pred == 1 else "HAM"})


# ----- EMAIL -----
@app.route("/predict_email", methods=["POST"])
def predict_email():
    data = request.get_json()
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    message = clean_text(message)
    X = email_tfidf.transform([message])
    pred = email_model.predict(X)[0]

    return jsonify({"prediction": "SPAM" if pred == 1 else "HAM"})


# ----- WHATSAPP (FIXED LOGIC) -----
@app.route("/predict_whatsapp", methods=["POST"])
def predict_whatsapp():
    data = request.get_json()
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    message = clean_text(message)
    X = whatsapp_tfidf.transform([message])
    pred_label = whatsapp_model.predict(X)[0]

    # ✅ Correct interpretation
    if pred_label in ["ham", "other"]:
        return jsonify({"prediction": "Not a scam"})
    else:
        return jsonify({"prediction": pred_label})


# ----- VOICE CALL -----
@app.route("/predict_voice", methods=["POST"])
def predict_voice():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.split(".")[-1]
    temp_path = f"temp_voice_{os.getpid()}.{ext}"
    file.save(temp_path)

    try:
        features = extract_voice_features(temp_path)
        scaled = voice_scaler.transform(features)
        pred = voice_model.predict(scaled)[0]
        label = voice_encoder.inverse_transform([pred])[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"prediction": label})


# ===========================
# RUN FLASK
# ===========================

if __name__ == "__main__":
    app.run(debug=True)
