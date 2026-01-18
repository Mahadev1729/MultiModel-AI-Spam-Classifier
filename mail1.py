import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1️⃣ Load dataset (change path if needed)
csv_path = "D:/project/dataset/CEAS_08.csv"

try:
    data = pd.read_csv(csv_path, encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv(csv_path, encoding="latin-1")

print("Columns in dataset:", data.columns)

# 2️⃣ Normalize column names (try to find label + text columns)

label_col = None
text_col = None

# Common column name options for label
possible_label_cols = ["label", "Label", "Email Type", "Category", "target", "class"]
# Common column name options for text
possible_text_cols = ["message", "text", "Email Text", "Body", "Message_body", "email", "text_combined", "body"]

for c in data.columns:
    if c in possible_label_cols and label_col is None:
        label_col = c
    if c in possible_text_cols and text_col is None:
        text_col = c

if label_col is None or text_col is None:
    raise ValueError(f"Could not find suitable label/text columns. Found cols: {list(data.columns)}")

print(f"Using label column: {label_col}")
print(f"Using text  column: {text_col}")

data = data[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})

print("\nSample rows:")
print(data.head())

# 3️⃣ Map labels to 0/1
# Adjust mapping according to your dataset labels
# Common: spam/phishing = 1, ham/legitimate = 0

data["label"] = data["label"].astype(str).str.lower().str.strip()

label_map = {
    "ham": 0,
    "legit": 0,
    "legitimate": 0,
    "safe": 0,
    "0": 0,

    "spam": 1,
    "phishing": 1,
    "phish": 1,
    "malicious": 1,
    "1": 1
}

data["label"] = data["label"].map(label_map)

# Drop rows where label could not be mapped
data = data.dropna(subset=["label"])
data["label"] = data["label"].astype(int)

# Drop missing text
data = data.dropna(subset=["text"])

X = data["text"]
y = data["label"]

print("\nClass distribution:")
print(y.value_counts())

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
svm_clf = LinearSVC()
svm_clf.fit(X_train_tfidf, y_train)

# 7️⃣ Evaluate
y_pred = svm_clf.predict(X_test_tfidf)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=["legit", "spam/phish"]))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8️⃣ Save model + vectorizer
joblib.dump(svm_clf, "svm_email_model1.pkl")
joblib.dump(tfidf, "tfidf_email_vectorizer1.pkl")

print("\n💾 Model saved as: svm_email_model.pkl")
print("💾 TF-IDF saved as: tfidf_email_vectorizer.pkl")

# 9️⃣ Test on some sample emails
sample_emails = [
    "Your PayPal account has been limited. Click here to verify your identity immediately.",
    "Hi team, please find attached the minutes of today's meeting.",
    "Congratulations! You have won a free iPhone. Submit your bank details to claim the prize.",
    "Reminder: Your electricity bill is due tomorrow. Please pay via the official portal."
]

sample_tfidf = tfidf.transform(sample_emails)
sample_pred = svm_clf.predict(sample_tfidf)

for msg, label in zip(sample_emails, sample_pred):
    print("\nEmail:", msg)
    print("Predicted:", "SPAM / PHISHING" if label == 1 else "LEGITIMATE")
