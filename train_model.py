import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# -----------------------
# Text Cleaning Function
# -----------------------
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("dataset/Resume.csv")

# Clean Resume Text
df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

X = df["cleaned_resume"]
y = df["Category"]

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# TF-IDF Vectorization
# -----------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------
# Train Model (Balanced)
# -----------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------
# Save Model
# -----------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n✅ Model trained & saved successfully!")