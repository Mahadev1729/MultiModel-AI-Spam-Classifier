import pandas as pd
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ============================================================
# 📌 1. TRAIN THE MODEL DIRECTLY
# ============================================================

DATASET = "D:/project/dataset/KAGGLE/DATASET-balanced.csv"

print("\n📌 Training model from:", DATASET)
df = pd.read_csv(DATASET)

if "LABEL" not in df.columns:
    raise ValueError("❌ ERROR: Dataset must have a column named 'LABEL'")

X = df.drop("LABEL", axis=1)
y = df["LABEL"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"\n🎯 MODEL TRAINED — Accuracy: {acc:.4f}")

joblib.dump(model, "voice.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")

print("\n💾 Model Saved: voice.pkl, scaler.pkl, encoder.pkl")


# ============================================================
# 📌 2. FEATURE EXTRACTION FROM AUDIO
# ============================================================

def get_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)

    feats = {
        "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "rms": np.mean(librosa.feature.rms(y=y)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y)),
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        feats[f"mfcc{i}"] = float(np.mean(mfcc[i-1]))

    return pd.DataFrame([feats])


# ============================================================
# 📌 3. PREDICT FROM MP3/WAV DIRECTLY (AUTO)
# ============================================================

def predict_audio(audio_file):
    print(f"\n🎧 Processing audio: {audio_file}")
    
    model = joblib.load("voice.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")

    feats = get_features(audio_file)
    scaled = scaler.transform(feats)
    pred = model.predict(scaled)[0]

    label = encoder.inverse_transform([pred])[0]

    print("\n🔮 FINAL RESULT →", label.upper())
    return label


# ============================================================
# 📌 4. GIVE YOUR AUDIO FILE HERE — ONLY THIS LINE CHANGES
# ============================================================

AUDIO_INPUT = "C:/Users/omsha/Downloads/audio-wav-16khz_62991_normalized.wav"   # CHANGE ONLY THIS 🔥

predict_audio(AUDIO_INPUT)
