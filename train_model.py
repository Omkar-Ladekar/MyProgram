import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("data.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)
print("\n📈 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.pkl")
print("\n✅ Model saved to 'model/model.pkl'")
