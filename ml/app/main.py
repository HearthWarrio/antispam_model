from fastapi import FastAPI
from pydantic import BaseModel
import os, json, joblib, sys, types

class SbertFeaturizer:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None, batch_size=128, normalize=True):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def transform(self, texts):
        self._ensure_model()
        return self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_model"] = None
        return d

class SpamPipeline:
    def __init__(self, featurizer: SbertFeaturizer, clf, threshold: float = 0.5):
        self.featurizer = featurizer
        self.clf = clf
        self.threshold = threshold

    def predict_proba(self, texts):
        import numpy as np
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
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    setattr(mod, "SbertFeaturizer", SbertFeaturizer)
    setattr(mod, "SpamPipeline",    SpamPipeline)
# ---------------------------------------------------------------------------

ART_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

app = FastAPI(title="Spam Classifier API", version="1.0.1")

class SpamRequest(BaseModel):
    text: str

class SpamBatchRequest(BaseModel):
    texts: list[str]

class CategoryRequest(BaseModel):
    text: str

class CategoryBatchRequest(BaseModel):
    texts: list[str]

def load_models():
    spam = None
    thr = 0.5
    try:
        spam = joblib.load(os.path.join(ART_DIR, "spam_clf.joblib"))
    except Exception as e:
        print("[load] spam model not loaded:", repr(e))
    tpath = os.path.join(ART_DIR, "spam_threshold.json")
    if os.path.exists(tpath):
        try:
            with open(tpath, "r", encoding="utf-8") as f:
                thr = json.load(f).get("threshold", 0.5)
        except Exception as e:
            print("[load] threshold read failed:", repr(e))
    cat = None
    try:
        cpath = os.path.join(ART_DIR, "category_clf.joblib")
        if os.path.exists(cpath):
            cat = joblib.load(cpath)
    except Exception as e:
        print("[load] category not loaded:", repr(e))
    return spam, thr, cat

SPAM_MODEL, THRESHOLD, CAT_MODEL = load_models()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "threshold": THRESHOLD,
        "has_spam_model": SPAM_MODEL is not None,
        "has_category": CAT_MODEL is not None,
        "artifacts_dir": ART_DIR,
    }

@app.post("/predict_spam")
def predict_spam(req: SpamRequest):
    if SPAM_MODEL is None:
        return {"error": "spam model is not available; train it first"}
    proba = float(SPAM_MODEL.predict_proba([req.text])[0][1])
    return {"spam_probability": proba, "is_spam": proba >= THRESHOLD, "threshold": THRESHOLD}

@app.post("/predict_spam_batch")
def predict_spam_batch(req: SpamBatchRequest):
    if SPAM_MODEL is None:
        return {"error": "spam model is not available; train it first"}
    probas = SPAM_MODEL.predict_proba(req.texts)[:, 1].tolist()
    preds = [p >= THRESHOLD for p in probas]
    return {"spam_probability": probas, "is_spam": preds, "threshold": THRESHOLD}

@app.post("/predict_category")
def predict_category(req: CategoryRequest):
    if CAT_MODEL is None:
        return {"error": "category model not available"}
    label = CAT_MODEL.predict([req.text])[0]
    return {"category": label}

@app.post("/predict_category_batch")
def predict_category_batch(req: CategoryBatchRequest):
    if CAT_MODEL is None:
        return {"error": "category model not available"}
    labels = CAT_MODEL.predict(req.texts).tolist()
    return {"category": labels}
