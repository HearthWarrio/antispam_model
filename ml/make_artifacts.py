import os, sys, types, json, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import matplotlib.pyplot as plt

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

class SbertFeaturizer:
    def __init__(self, model_name="all-MiniLM-L6-v2", device="auto", batch_size=128, normalize=True):
        self.model_name = model_name
        self.device = device   # 'auto'|'cpu'|'cuda'
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _resolve_device(self):
        try:
            import torch
            has_cuda = bool(getattr(torch, "version", None) and getattr(torch.version, "cuda", None)) and torch.cuda.is_available()
        except Exception:
            has_cuda = False
        if self.device == "cuda" and not has_cuda:
            return "cpu"
        if self.device in ("auto", None):
            return "cuda" if has_cuda else "cpu"
        return self.device

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            dev = self._resolve_device()
            self._model = SentenceTransformer(self.model_name, device=dev)

    def transform(self, texts):
        self._ensure_model()
        return self._model.encode(
            list(texts), batch_size=self.batch_size,
            convert_to_numpy=True, normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    def __getstate__(self):
        d = dict(self.__dict__); d["_model"] = None; return d


class SpamPipeline:
    def __init__(self, featurizer: SbertFeaturizer, clf, threshold: float = 0.5):
        self.featurizer = featurizer
        self.clf = clf
        self.threshold = threshold
    def predict_proba(self, texts):
        X = self.featurizer.transform(texts)
        proba = self.clf.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - proba, proba])
    def predict(self, texts):
        p = self.predict_proba(texts)[:, 1]
        return (p >= self.threshold).astype(int)

m = sys.modules.get("__main__")
if m is not None:
    setattr(m, "SbertFeaturizer", SbertFeaturizer)
    setattr(m, "SpamPipeline",    SpamPipeline)
for name in ("train_gpu", "src.train_gpu"):
    mod = sys.modules.get(name) or types.ModuleType(name)
    setattr(mod, "SbertFeaturizer", SbertFeaturizer)
    setattr(mod, "SpamPipeline",    SpamPipeline)
    sys.modules[name] = mod

DATA_CSV = Path("data/processed/combined.csv")
ART_DIR  = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_CSV.exists(), f"Не найден датасет: {DATA_CSV}"
assert (ART_DIR/"spam_clf.joblib").exists(), "Не найден artifacts/spam_clf.joblib. Обучите модель."
assert (ART_DIR/"spam_threshold.json").exists(), "Не найден artifacts/spam_threshold.json."

df = pd.read_csv(DATA_CSV)
X = df["text"].astype(str).values
y = df["label"].astype(int).values

pipe = joblib.load(ART_DIR/"spam_clf.joblib")
with open(ART_DIR/"spam_threshold.json","r",encoding="utf-8") as f:
    thr = float(json.load(f).get("threshold", 0.5))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

proba = pipe.predict_proba(Xte)[:,1]
ypred = (proba >= thr).astype(int)

p, r, f1, _ = precision_recall_fscore_support(yte, ypred, average="binary")
ap = average_precision_score(yte, proba)

report = {"threshold": float(thr), "precision": float(p), "recall": float(r), "f1": float(f1),
          "AP": float(ap), "support": int(len(yte))}
with open(ART_DIR/"report.json","w",encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("== Итоговые метрики (holdout 15%) =="); print(report)

prec, rec, _ = precision_recall_curve(yte, proba)
plt.figure(); plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve")
plt.savefig(ART_DIR/"pr_curve.png", dpi=150, bbox_inches="tight"); plt.close()

cm = confusion_matrix(yte, ypred)
plt.figure(); plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix"); plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.xticks([0,1], ["Ham","Spam"]); plt.yticks([0,1], ["Ham","Spam"])
plt.savefig(ART_DIR/"confusion_matrix.png", dpi=150, bbox_inches="tight"); plt.close()

bal = pd.Series(y).value_counts().sort_index().rename(index={0:"Ham",1:"Spam"})
plt.figure(); bal.plot(kind="bar"); plt.title("Class balance (full dataset)"); plt.ylabel("count")
plt.savefig(ART_DIR/"class_balance.png", dpi=150, bbox_inches="tight"); plt.close()

df_hold = pd.DataFrame({"text":Xte, "y_true":yte, "proba":proba, "pred":ypred})
def shorten(s, n=160): s=" ".join(s.split()); return s[:n]+"…" if len(s)>n else s
fp = df_hold[(df_hold.y_true==0)&(df_hold.pred==1)].sort_values("proba", ascending=False).head(10)
fn = df_hold[(df_hold.y_true==1)&(df_hold.pred==0)].sort_values("proba", ascending=True).head(10)
fp[["text","proba"]].assign(text=lambda d: d["text"].map(shorten)).to_csv(ART_DIR/"errors_fp_top.csv", index=False, encoding="utf-8")
fn[["text","proba"]].assign(text=lambda d: d["text"].map(shorten)).to_csv(ART_DIR/"errors_fn_top.csv", index=False, encoding="utf-8")

print("\nСохранено в artifacts/: report.json, pr_curve.png, confusion_matrix.png, class_balance.png, errors_fp_top.csv, errors_fn_top.csv")
