# nlp/21_build_review_embeddings.py
# Build per-review embeddings WITH `sku`, ready for best-review scoring.
# Input : data/processed/reviews_merged.parquet  (id, sku, ts, stars, text, source)
# Output: data/processed/reviews_with_embeddings.parquet
# Env   : EMB_REVIEW_MODEL (default "BAAI/bge-small-en-v1.5"), BATCH (default 256)

from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA = Path("data/processed")
INP  = DATA / "reviews_merged.parquet"
OUT  = DATA / "reviews_with_embeddings.parquet"

MODEL_NAME = os.environ.get("EMB_REVIEW_MODEL", "BAAI/bge-small-en-v1.5")
BATCH      = int(os.environ.get("BATCH", "256"))
MIN_LEN    = 10          # drop ultra-short reviews
MAX_LEN    = 2000        # truncate very long reviews before embedding (tokens handled by model anyway)
DEDUP      = True        # disable if you truly want raw duplicates
LOWERCASE  = True

URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PROMO_RE   = re.compile(r"(discount code|use code|sponsored|i received this.*free)", re.IGNORECASE)
REPEAT_RE  = re.compile(r"(.)\1{9,}")  # 10+ repeated characters like "!!!!!!!!!"

def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_parquet(path)
    need = {"id","sku","text"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    # keep only useful columns
    keep = [c for c in ["id","sku","ts","stars","text","source"] if c in df.columns]
    df = df[keep].copy()
    # coerce types
    df["id"] = df["id"].astype(str)
    df["sku"] = df["sku"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    if "stars" in df.columns:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    return df

def basic_clean(s: str) -> str:
    s = s.strip()
    if LOWERCASE:
        s = s.lower()
    # whitespace normalize
    s = re.sub(r"\s+", " ", s)
    return s

def looks_spammy(s: str) -> bool:
    url_count = len(URL_RE.findall(s))
    if url_count >= 2:
        return True
    if PROMO_RE.search(s):
        return True
    if REPEAT_RE.search(s):
        return True
    return False

def filter_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # length filter
    df = df[df["text"].str.len() >= MIN_LEN].copy()
    if df.empty:
        return df
    # spam filter
    df["__spam"] = df["text"].apply(looks_spammy)
    df = df[~df["__spam"]].drop(columns="__spam")
    if df.empty:
        return df
    # cleaned text (for dedup + truncation)
    df["__clean"] = df["text"].apply(basic_clean).str.slice(0, MAX_LEN)
    if DEDUP:
        # dedup within (sku, clean_text)
        before = len(df)
        df = df.drop_duplicates(subset=["sku","__clean"])
        after = len(df)
        print(f"[dedup] removed {before-after:,} duplicate (sku,text) rows")
    return df

def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine via dot
    ).astype(np.float32)

def main():
    print(f"[load] {INP}")
    df = load_reviews(INP)
    n0 = len(df)
    print(f"[rows] {n0:,}")

    df = filter_and_prepare(df)
    if df.empty:
        raise RuntimeError("No reviews left after filtering. Relax filters and retry.")

    print(f"[model] {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[embed] encoding {len(df):,} reviews (batch={BATCH})…")
    vecs = embed_texts(model, df["__clean"].tolist(), BATCH)

    # assemble output
    out = df.drop(columns=["__clean"], errors="ignore").copy()
    out["embedding"] = list(vecs)  # each row: np.ndarray(float32, dim)

    # persist
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    dim = vecs.shape[1]
    kept = len(out)
    print(f"[ok] wrote {OUT}  rows={kept:,}  dim={dim}")
    # quick peek
    cols = [c for c in ["id","sku","stars","text"] if c in out.columns]
    print("[sample]\n", out[cols].head(3).to_string(index=False))

if __name__ == "__main__":
    main()
