
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

# Load dataset -  replace with your file path
df = pd.read_csv("/Users/emmanuelkaribiye/Fake-news-detection/data.csv")

# Basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.strip()
    return text

# Apply cleaning
df['cleaned_headlines'] = df['headlines'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_headlines'], df['outcome'], test_size=0.2, random_state=42, stratify=df['outcome'])

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB()
}

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

# Train and evaluate each model
results = {}

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(report)
    print(f"AUC-ROC: {auc:.3f}")
