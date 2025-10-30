#!/usr/bin/env python
import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

RANDOM_STATE = 42

def tune_threshold(y_true, proba, target="precision", min_recall=0.8):
    precision, recall, thresh = precision_recall_curve(y_true, proba)
    best_t, best_prec, best_rec, best_f1 = 0.5, 0.0, 0.0, 0.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresh):
        if r >= min_recall:
            f1 = 2*p*r/(p+r+1e-9)
            if target == "precision":
                key = p
            elif target == "f1":
                key = f1
            else:
                key = r
            if key > (best_prec if target=="precision" else best_f1 if target=="f1" else best_rec):
                best_t, best_prec, best_rec, best_f1 = t, p, r, f1
    return max(0.5, best_t), best_prec, best_rec, best_f1

def train_tfidf_logreg(df: pd.DataFrame, artifacts_dir: Path):
    X = df["text"].values
    y = df["label"].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1,2), max_df=0.9, min_df=2, max_features=200000)
    clf = LogisticRegression(max_iter=2000, n_jobs=4, class_weight="balanced")
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)
    proba = calibrated.predict_proba(X_te)[:,1]
    thr, p, r, f1 = tune_threshold(y_te, proba, target="precision", min_recall=0.85)
    y_pred = (proba >= thr).astype(int)
    pr, rc, f1s, _ = precision_recall_fscore_support(y_te, y_pred, average="binary")
    ap = average_precision_score(y_te, proba)
    print("[final] threshold=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  AP=%.4f" % (thr, pr, rc, f1s, ap))
    print(classification_report(y_te, y_pred, digits=4))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, artifacts_dir / "spam_clf.joblib")
    joblib.dump({"threshold": float(thr)}, artifacts_dir / "spam_threshold.json")
    plt.figure(); 
    precision, recall, _ = precision_recall_curve(y_te, proba); 
    import matplotlib.pyplot as plt
    plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve"); 
    plt.savefig(artifacts_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    cm = confusion_matrix(y_te, y_pred); plt.figure(); plt.imshow(cm); plt.title("Confusion Matrix"); plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.xticks([0,1], ["Ham","Spam"]); plt.yticks([0,1], ["Ham","Spam"])
    plt.savefig(artifacts_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    with open(artifacts_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": float(thr), "precision": float(pr), "recall": float(rc), "f1": float(f1s), "AP": float(ap)}, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    train_tfidf_logreg(df, Path(args.artifacts_dir))

if __name__ == "__main__":
    main()
