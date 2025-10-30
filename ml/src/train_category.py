#!/usr/bin/env python
import argparse, re, json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

SEED_RULES = {
    "ads": [r"sale|discount|% off|buy now|subscribe|offer|limited time|продажа|скидк|распродаж"],
    "jobs": [r"vacancy|position|hiring|cv|resume|salary|ваканси|зарплат|резюме|работа|приглашаем"],
    "dating": [r"date|meet singles|relationship|love|знакомств|свидани|любов"],
    "finance": [r"loan|credit|mortgage|debt|investment|crypto|кред|ипотек|долг|инвест|крипт"],
    "support": [r"ticket|support|help desk|issue|password|поддержк|парол|тикет|заявка"],
    "personal": [r"hi|hello|how are you|see you|thanks|привет|здравств|спасибо|как дела|увидимс"],
}

def weak_label(text: str):
    low = str(text).lower()
    for lab, pats in SEED_RULES.items():
        for pat in pats:
            if re.search(pat, low):
                return lab
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/combined.csv")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["weak_label"] = df["text"].apply(weak_label).fillna("unknown")
    seed = df[df["weak_label"] != "unknown"]
    X_tr, X_te, y_tr, y_te = train_test_split(seed["text"], seed["weak_label"], test_size=0.2, random_state=42, stratify=seed["weak_label"])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.95, max_features=200000)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=4, class_weight="balanced", multi_class="auto"))
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    print(classification_report(y_te, y_pred, digits=4))

    art = Path(args.artifacts_dir); art.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, art / "category_clf.joblib")
    with open(art / "category_report.json", "w", encoding="utf-8") as f:
        json.dump({"labels": sorted(seed["weak_label"].unique().tolist()), "train": len(X_tr), "test": len(X_te)}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
