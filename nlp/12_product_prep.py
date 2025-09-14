# nlp/12_product_prep.py
# Builds:
#   - data/processed/product_bm25.pkl  (tokenized corpus + SKU order)
#   - data/processed/topic_vecs.parquet (if topic files exist)
#
# Inputs (one of):
#   - data/processed/products.parquet            (sku, agg_text, n_reviews, avg_stars, last_ts)
#   - data/processed/product_emb_meta.parquet    (sku, agg_text, n_reviews, avg_stars, last_ts)
#
# Optional inputs:
#   - data/processed/topics_named_llm.parquet or topics_named.parquet (topic_id, topic_label)
#   - data/processed/topic_cards.parquet (topic_id, topic_label?, headline?, summary?)
#
# Notes:
#   - No GPU needed; topic embedding is tiny and runs on CPU.

from __future__ import annotations
import os, re, pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # only needed if topic files exist

DATA = Path("data/processed")
PROD_TXT1 = DATA / "products.parquet"
PROD_TXT2 = DATA / "product_emb_meta.parquet"

BM25_OUT = DATA / "product_bm25.pkl"
TOPIC_VECS_OUT = DATA / "topic_vecs.parquet"

TOPICS1 = DATA / "topics_named_llm.parquet"
TOPICS2 = DATA / "topics_named.parquet"
CARDS   = DATA / "topic_cards.parquet"

TOPIC_MODEL = os.environ.get("TOPIC_MODEL", "BAAI/bge-small-en-v1.5")

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOP = {
    # minimal english stoplist
    "a","an","and","the","is","are","am","be","been","to","for","of","in","on","at","by",
    "it","its","this","that","with","from","as","or","if","but","than","then","so",
    "i","you","he","she","we","they","my","your","our","their","me","him","her","us","them",
    "was","were","will","would","should","could","may","might","can","cannot","cant","won't",
}

def log(msg: str): print(msg, flush=True)

def pick_products_path() -> Path:
    if PROD_TXT1.exists(): return PROD_TXT1
    if PROD_TXT2.exists(): return PROD_TXT2
    raise FileNotFoundError("products.parquet or product_emb_meta.parquet not found in data/processed/")

def load_products(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "sku" not in df.columns: raise ValueError(f"{path} must have column 'sku'")
    # pick text column
    txt_col = "agg_text" if "agg_text" in df.columns else None
    if not txt_col:
        # try other names
        for c in ["text","merged_text","description"]:
            if c in df.columns: txt_col = c; break
    if not txt_col:
        raise ValueError(f"No text column found in {path} (expect 'agg_text').")
    need = ["sku", txt_col]
    keep = [c for c in need if c in df.columns]
    df = df[keep].copy()
    df[txt_col] = df[txt_col].fillna("").astype(str)
    return df.rename(columns={txt_col: "agg_text"})

def tokenize(s: str) -> List[str]:
    s = s.lower()
    toks = [t for t in TOKEN_RE.findall(s) if t not in STOP and len(t) > 1]
    return toks[:5000]  # safety cap

def build_bm25_corpus(df: pd.DataFrame) -> Tuple[List[List[str]], List[str]]:
    corpus = [tokenize(t) for t in df["agg_text"].tolist()]
    skus = df["sku"].astype(str).tolist()
    return corpus, skus

def save_bm25(corpus: List[List[str]], skus: List[str]):
    BM25_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_OUT, "wb") as f:
        pickle.dump({"skus": skus, "corpus": corpus, "tokenizer": "simple_en_v1"}, f, protocol=4)
    log(f"[ok] wrote {BM25_OUT}  docs={len(skus):,}")

def coalesce_topics() -> pd.DataFrame | None:
    # try named topics
    for p in [TOPICS1, TOPICS2]:
        if p.exists():
            df = pd.read_parquet(p)
            if "topic_id" not in df.columns:
                for c in ["topic","cluster","label_id","topicid","topicId"]:
                    if c in df.columns: df = df.rename(columns={c:"topic_id"}); break
            if "topic_label" not in df.columns:
                for c in ["label","name","title","topic_name","card_title"]:
                    if c in df.columns: df = df.rename(columns={c:"topic_label"}); break
            if "topic_id" in df.columns:
                df = df[["topic_id","topic_label"]].drop_duplicates("topic_id")
                return df
    return None

def build_topic_texts() -> pd.DataFrame | None:
    tn = coalesce_topics()
    if tn is None and not CARDS.exists():
        return None

    # Start from topic names if present
    rows = []
    if tn is not None:
        rows = [{"topic_id": int(r.topic_id), "topic_label": str(r.topic_label or "")} for _, r in tn.iterrows()]

    # Merge in cards if present
    if CARDS.exists():
        cards = pd.read_parquet(CARDS)
        if "topic_id" not in cards.columns:
            return pd.DataFrame(rows) if rows else None
        # friendly fields
        label = "topic_label" if "topic_label" in cards.columns else None
        headline = "headline" if "headline" in cards.columns else None
        summary  = "summary"  if "summary"  in cards.columns else None
        cards = cards[["topic_id"] + [c for c in [label, headline, summary] if c]].copy()
        cards = cards.groupby("topic_id").first().reset_index()
        base = pd.DataFrame(rows) if rows else cards[["topic_id"]].copy()
        out = base.merge(cards, on="topic_id", how="outer")
    else:
        out = pd.DataFrame(rows)

    # Compose text
    def compose(row):
        parts = []
        for c in ["topic_label","headline","summary"]:
            if c in row and isinstance(row[c], str) and row[c].strip():
                parts.append(row[c].strip())
        return " ".join(parts)[:2000] if parts else ""
    out["topic_text"] = out.apply(compose, axis=1)
    out = out[out["topic_text"].str.len() >= 3].copy()
    return out

def embed_topics(df: pd.DataFrame) -> pd.DataFrame:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed; cannot embed topics.")
    model = SentenceTransformer(TOPIC_MODEL, device="cpu")  # tiny workload; CPU is fine
    vecs = model.encode(df["topic_text"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    out = df.copy()
    out["embedding"] = list(np.asarray(vecs, dtype=np.float32))
    return out

def main():
    prod_path = pick_products_path()
    log(f"[load] {prod_path}")
    prods = load_products(prod_path)
    log(f"[bm25] tokenizing {len(prods):,} product texts…")
    corpus, skus = build_bm25_corpus(prods)
    save_bm25(corpus, skus)

    # Topics (optional)
    tops = build_topic_texts()
    if tops is None or tops.empty:
        log("[topic] no topic files found; skipping topic vectors.")
        return
    log(f"[topic] embedding {len(tops):,} topics with {TOPIC_MODEL}…")
    tv = embed_topics(tops)
    TOPIC_VECS_OUT.parent.mkdir(parents=True, exist_ok=True)
    tv.to_parquet(TOPIC_VECS_OUT, index=False)
    log(f"[ok] wrote {TOPIC_VECS_OUT} (rows={len(tv):,}, dim={len(tv['embedding'].iloc[0])})")

if __name__ == "__main__":
    main()
