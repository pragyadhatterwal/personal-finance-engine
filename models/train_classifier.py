# models/train_classifier.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

DATA_PATH = os.path.join("data", "transactions_sample.csv")
MODEL_PATH = os.path.join("models", "category_model.pkl")

def combine_text_columns(df: pd.DataFrame) -> pd.Series:
    """
    Combine text fields to form a richer feature space.
    """
    merchant = df["merchant"].fillna("")
    raw = df["raw"].fillna("")
    tx_type = df["type"].fillna("")
    return (merchant + " " + raw + " " + tx_type).astype(str)

def main():
    df = pd.read_csv(DATA_PATH)
    # Keep only rows with category labels
    df = df.dropna(subset=["category"])
    if df.empty:
        raise ValueError("No labeled data found. Please label 'category' in the CSV.")

    X = df[["merchant", "raw", "type"]]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    text_features = FunctionTransformer(combine_text_columns, validate=False)
    pipeline = Pipeline(steps=[
        ("text", text_features),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
