#!/usr/bin/env python
import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, precision_recall_curve, average_precision_score

from xgboost import XGBClassifier

# SBERT
from sentence_transformers import SentenceTransformer

RANDOM_STATE = 42

def tune_threshold(y_true, proba, target="precision", min_recall=0.8):
    precision, recall, thresh = precision_recall_curve(y_true, proba)
    best_t, best_prec, best_rec, best_f1 = 0.5, 0.0, 0.0, 0.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresh):
        if r >= min_recall:
            f1 = 2*p*r/(p+r+1e-9)
            key = p if target=="precision" else f1 if target=="f1" else r
            if key > (best_prec if target=="precision" else best_f1 if target=="f1" else best_rec):
                best_t, best_prec, best_rec, best_f1 = t, p, r, f1
    return best_t, best_prec, best_rec, best_f1

class SbertFeaturizer:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None, batch_size=128, normalize=True):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None  # loaded lazily

    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def transform(self, texts):
        self._ensure_model()
        emb = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        return emb

    # make pickle smaller
    def __getstate__(self):
        d = dict(self.__dict__)
        d["_model"] = None
        return d

class SpamPipeline:
    def __init__(self, featurizer: SbertFeaturizer, clf: XGBClassifier, threshold: float = 0.5):
        self.featurizer = featurizer
        self.clf = clf
        self.threshold = threshold

    def predict_proba(self, texts):
        X = self.featurizer.transform(texts)
        proba = self.clf.predict_proba(X)[:,1]
        return np.column_stack([1.0 - proba, proba])

    def predict(self, texts):
        proba = self.predict_proba(texts)[:,1]
        return (proba >= self.threshold).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/combined.csv")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    ap.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if None)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--min_recall", type=float, default=0.85)
    ap.add_argument("--target_metric", type=str, default="precision", choices=["precision","f1","recall"])
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = df["text"].astype(str).values
    y = df["label"].astype(int).values

    # split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)

    # features (GPU if torch+cuda installed; sentence-transformers picks device automatically)
    feat = SbertFeaturizer(model_name=args.embedder, device=args.device, batch_size=args.batch_size, normalize=True)
    Xtr = feat.transform(X_tr)
    Xte = feat.transform(X_te)

    # XGBoost with GPU
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        eval_metric="logloss",
        n_jobs=0,
    )
    clf.fit(Xtr, y_tr)

    proba = clf.predict_proba(Xte)[:,1]
    thr, p, r, f1 = tune_threshold(y_te, proba, target=args.target_metric, min_recall=args.min_recall)
    y_pred = (proba >= thr).astype(int)

    pr, rc, f1s, _ = precision_recall_fscore_support(y_te, y_pred, average="binary")
    ap_score = average_precision_score(y_te, proba)
    print("[GPU final] threshold=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  AP=%.4f" % (thr, pr, rc, f1s, ap_score))
    print(classification_report(y_te, y_pred, digits=4))

    # Save pipeline compatible with API
    pipe = SpamPipeline(featurizer=feat, clf=clf, threshold=float(thr))

    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, art / "spam_clf.joblib")
    with open(art / "spam_threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": float(thr)}, f, ensure_ascii=False, indent=2)
    print("[saved] artifacts to", art)

if __name__ == "__main__":
    main()
