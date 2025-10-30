#!/usr/bin/env python
import argparse
from pathlib import Path
import urllib.request

DATASETS = {
    "sms": ["https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"],
    "spamassassin": [
        "https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20021010_spam.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tgz",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tgz",
    ],
    "ling": [
        "http://csmining.org/index.php/ling-spam-datasets.html"
    ]
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def fetch(url, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = url.split("/")[-1]
    out = out_dir / fname
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req) as r:
            data = r.read()
            ctype = r.headers.get("Content-Type", "").lower()
        if ("text/html" in ctype) or len(data) < 1024:
            print("[warn] got HTML/small file, skipping:", url)
            return
        with open(out, "wb") as f:
            f.write(data)
        print("[ok]", out, f"({len(data)//1024} KB)")
    except Exception as e:
        print("[warn] failed", url, e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="sms,spamassassin,ling")
    ap.add_argument("--out", type=str, default="data/raw")
    args = ap.parse_args()

    targets = [t.strip() for t in args.datasets.split(",") if t.strip()]
    root = Path(args.out)
    for t in targets:
        for url in DATASETS.get(t, []):
            fetch(url, root / t)
    print("[done]")

if __name__ == "__main__":
    main()
