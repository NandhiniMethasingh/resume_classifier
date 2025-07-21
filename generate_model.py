# generate_model.py — Rebuilds classifier & vectorizer for your version

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load your features.xlsx file
df = pd.read_excel("features_cleaned.xlsx")
X_text = df["processed_text"]
y = df["label"]

# Step 2: Convert resume text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_text)

# Step 3: Train-test split (use only training data for saving)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train a Random Forest model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Step 5: Save model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(classifier, "model/classifier.pkl")

print("Model and vectorizer saved successfully to 'model/' folder!")
