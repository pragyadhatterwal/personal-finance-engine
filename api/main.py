# api/main.py

import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

MODEL_PATH = os.path.join("models", "category_model.pkl")

app = FastAPI(title="Finance Insight Engine API")

class Txn(BaseModel):
    merchant: Optional[str] = ""
    raw: Optional[str] = ""
    type: Optional[str] = "debit"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify")
def classify(txn: Txn):
    if not os.path.exists(MODEL_PATH):
        return {"category": None, "confidence": None, "message": "Model not found. Train it first."}
    clf = joblib.load(MODEL_PATH)
    X = [{"merchant": txn.merchant or "", "raw": txn.raw or "", "type": txn.type or ""}]
    try:
        proba = clf.predict_proba(X)[0]
        pred = clf.predict(X)[0]
        # get confidence
        classes = clf.classes_
        conf = float(proba[list(classes).index(pred)])
        return {"category": pred, "confidence": round(conf, 4)}
    except Exception as e:
        return {"category": None, "confidence": None, "error": str(e)}
