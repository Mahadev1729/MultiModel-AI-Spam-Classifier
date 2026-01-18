import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib   # ← REQUIRED for saving model

# 1️⃣ Load dataset
try:
    data = pd.read_csv("D:/project/dataset/spam.csv", encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv("D:/project/dataset/spam.csv", encoding="latin-1")

# Show columns
print("Columns:", data.columns)

# Fix column names if needed
if 'label' not in data.columns or 'message' not in data.columns:
    if {'v1', 'v2'}.issubset(data.columns):
        data = data.rename(columns={'v1': 'label', 'v2': 'message'})
    else:
        raise ValueError(f"Expected 'label' and 'message' not found. Found: {list(data.columns)}")

print(data.head())

# Clean and map labels
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
data.dropna(inplace=True)

X = data['message']
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train))
print("Test size :", len(X_test))

# TF-IDF vectorizer
tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=5000
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF shape (train):", X_train_tfidf.shape)

# Train SVM model
svm_clf = LinearSVC()
svm_clf.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = svm_clf.predict(X_test_tfidf)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=['ham', 'spam']))
print("\n🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 8️⃣ ⭐ SAVE MODEL + TF-IDF ⭐
joblib.dump(svm_clf, "svm_sms_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\n💾 Model saved as: svm_sms_model.pkl")
print("💾 TF-IDF saved as: tfidf_vectorizer.pkl")


# 7️⃣ Example predictions
sample_messages = [
    "Congratulations! You have won ₹10,00,000. Click the link to claim your prize!",
    "Bro are we meeting at 5pm today?",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
]

sample_tfidf = tfidf.transform(sample_messages)
sample_pred = svm_clf.predict(sample_tfidf)

for msg, label in zip(sample_messages, sample_pred):
    print("\nMessage:", msg)
    print("Predicted:", "SPAM" if label == 1 else "HAM")
