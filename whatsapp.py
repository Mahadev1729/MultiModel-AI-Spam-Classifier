import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1️⃣ Load dataset
csv_path = "D:/project/dataset/Dataset_5971.csv"  # change if file name is different

try:
    data = pd.read_csv(csv_path, encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv(csv_path, encoding="latin-1")

print("Columns in dataset:", list(data.columns))

# Dataset description says columns: LABEL, TEXT, URL, EMAIL, PHONE
# We will use LABEL and TEXT only :contentReference[oaicite:1]{index=1}
label_col = "LABEL"
text_col = "TEXT"

if label_col not in data.columns or text_col not in data.columns:
    raise ValueError(f"Expected columns 'LABEL' and 'TEXT'. Found: {list(data.columns)}")

# Keep only needed columns
data = data[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})

print("\nSample rows:")
print(data.head())

# 2️⃣ Basic cleaning
data.dropna(subset=["label", "text"], inplace=True)

data["label"] = data["label"].astype(str).str.strip().str.lower()
data["text"] = data["text"].astype(str).str.strip()

print("\nLabel distribution:")
print(data["label"].value_counts())

# At this point labels should be: "ham", "spam", "smishing"

# 3️⃣ Feature & label variables
X = data["text"]
y = data["label"]   # multi-class: ham / spam / smishing

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))

# 5️⃣ TF-IDF vectorizer
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF shape (train):", X_train_tfidf.shape)

# 6️⃣ Train SVM model
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# 7️⃣ Evaluate
y_pred = model.predict(X_test_tfidf)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=["ham", "spam", "smishing"]))

# 8️⃣ Save model + vectorizer
joblib.dump(model, "svm_phishing_sms_model.pkl")
joblib.dump(tfidf, "tfidf_phishing_sms_vectorizer.pkl")

print("\n💾 Saved:")
print("  - svm_phishing_sms_model.pkl")
print("  - tfidf_phishing_sms_vectorizer.pkl")

# 9️⃣ Helper to print human-friendly result
def explain_prediction(message: str) -> str:
    vec = tfidf.transform([message])
    pred_label = model.predict(vec)[0]   # "ham", "spam", or "smishing"

    if pred_label == "ham":
        return "Not a scam message"
    else:
        # spam or smishing
        return f"Scam message – Type: {pred_label}"

# 🔟 Test on some sample messages
sample_messages = [
    "Congratulations! You have won a car. Click this link to claim your prize now.",
    "588508 is your OTP to access DigiLocker. OTP is confidential and valid for 10 minutes. For security reasons, DO NOT share this OTP with anyone.",
    "Dear customer, your bank account has been blocked. Please click here to verify your details.",
]

print("\n🔍 Sample predictions:")
for msg in sample_messages:
    print("\nMessage:", msg)
    print("Result :", explain_prediction(msg))