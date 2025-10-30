#!/usr/bin/env python
import argparse, re, tarfile, zipfile
from pathlib import Path
import pandas as pd

MAGIC = {
    "gz": b"\x1f\x8b",
    "bz2": b"BZ",
}

def is_probably_tar(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 100:
            return False
        with open(path, "rb") as f:
            head = f.read(4)
        return tarfile.is_tarfile(path) or head.startswith(MAGIC["gz"]) or head.startswith(MAGIC["bz2"])
    except Exception:
        return False

def normalize_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = re.sub(r"(?im)^(from|to|subject|reply-to|cc|bcc|date):.*$", " ", txt)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"https?://\S+|www\.\S+", " URL ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def parse_sms(path: Path):
    rows = []
    for zf in path.glob("*.zip"):
        try:
            with zipfile.ZipFile(zf, "r") as z:
                for name in z.namelist():
                    if name.endswith("SMSSpamCollection"):
                        with z.open(name) as f:
                            for line in f:
                                parts = line.decode("utf-8", errors="ignore").rstrip("\n").split("\t", 1)
                                if len(parts) == 2:
                                    label = 1 if parts[0].lower() == "spam" else 0
                                    rows.append((normalize_text(parts[1]), label, "sms"))
        except Exception as e:
            print(f"[warn] bad zip skipped: {zf.name} ({e})")
    return pd.DataFrame(rows, columns=["text", "label", "source"])

def extract_tar_into(tmp: Path, tar_path: Path):
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, mode="r:*") as tf:
            tf.extractall(tmp)
        return True
    except Exception as e:
        print(f"[warn] tar extract failed: {tar_path.name} ({e})")
        return False

def read_text(p: Path):
    for enc in ("utf-8", "latin-1", "cp1251"):
        try:
            return p.read_text(enc)
        except Exception:
            continue
    return ""

def parse_spamassassin(path: Path):
    rows = []
    tmp = path / "_x"
    tarballs = list(path.glob("*.tar.bz2")) + list(path.glob("*.tgz")) + list(path.glob("*.tar.gz"))
    for tarball in tarballs:
        if not is_probably_tar(tarball):
            print(f"[skip] not an archive (maybe HTML/403): {tarball.name}")
            continue
        extract_tar_into(tmp, tarball)
    for p in tmp.rglob("*"):
        if p.is_file():
            txt = read_text(p)
            if not txt.strip():
                continue
            label = 1 if ("spam" in str(p).lower() and "ham" not in str(p).lower()) else 0
            rows.append((normalize_text(txt), label, "spamassassin"))
    return pd.DataFrame(rows, columns=["text", "label", "source"])

def parse_ling(path: Path):
    rows = []
    for cls in ["spam", "legit"]:
        for p in path.rglob(f"*{cls}*/*"):
            if p.is_file():
                txt = read_text(p)
                if not txt:
                    continue
                label = 1 if cls == "spam" else 0
                rows.append((normalize_text(txt), label, "ling"))
    return pd.DataFrame(rows, columns=["text", "label", "source"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out", type=str, default="data/processed/combined.csv")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    dfs = []
    sms = parse_sms(raw / "sms")
    if not sms.empty: dfs.append(sms)
    sa  = parse_spamassassin(raw / "spamassassin")
    if not sa.empty: dfs.append(sa)
    ling = parse_ling(raw / "ling")
    if not ling.empty: dfs.append(ling)

    if not dfs:
        raise SystemExit("Нет данных. Скачайте их через scripts/download_data.py и распакуйте где нужно.")
    df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"]).drop_duplicates(subset=["text"])
    df = df[(df["text"].str.len() > 5) & (df["text"].str.len() < 50000)]
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("[ok]", out, "rows:", len(df)); print(df["label"].value_counts())

if __name__ == "__main__":
    main()
