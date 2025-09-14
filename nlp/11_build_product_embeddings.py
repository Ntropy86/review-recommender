# nlp/11_build_product_embeddings.py
# Build embeddings for PRODUCTS or REVIEWS with progress, sharding, and resume.
from __future__ import annotations
import argparse, os, re, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq

DATA = Path("data/processed")
PROD_IN   = DATA / "products.parquet"
REV_IN    = DATA / "reviews_merged.parquet"
PROD_OUT_EMB  = DATA / "product_emb.npy"
PROD_OUT_META = DATA / "product_emb_meta.parquet"
REV_OUT       = DATA / "reviews_with_embeddings.parquet"

EMB_MODEL_DEFAULT = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")
BATCH_DEFAULT     = int(os.environ.get("BATCH", "256"))

MIN_TEXT_LEN = 10
MAX_TEXT_LEN = 4000

URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PROMO_RE = re.compile(r"(discount code|use code|sponsored|i received this.*free)", re.IGNORECASE)
REPEAT_RE= re.compile(r"(.)\1{9,}")

def log(msg: str):
    print(msg, flush=True)

def normalize_text(s: str) -> str:
    if not isinstance(s, str): s = "" if s is None else str(s)
    s = s.replace("\r"," ").replace("\n"," ").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:MAX_TEXT_LEN]

def looks_spammy(s: str) -> bool:
    return (len(URL_RE.findall(s)) >= 2) or bool(PROMO_RE.search(s)) or bool(REPEAT_RE.search(s))

def get_model(name: str, device: str | None):
    log(f"[model] loading '{name}' (device={device or 'auto'})")
    model = SentenceTransformer(name, device=device)
    return model

def encode_shard(model: SentenceTransformer, texts: list[str], batch: int) -> np.ndarray:
    return model.encode(texts, batch_size=batch, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)

# ---------------- PRODUCT ----------------
def build_product_embeddings(args):
    src = Path(args.input or PROD_IN)
    if not src.exists():
        raise FileNotFoundError(f"Product input not found: {src}")
    df = pd.read_parquet(src)
    if "sku" not in df.columns: raise ValueError(f"{src} must have 'sku'")
    txt_col = args.text_col or ("agg_text" if "agg_text" in df.columns else None)
    if not txt_col: raise ValueError("Provide --text-col for product text (e.g., agg_text).")
    df = df.copy()
    df[txt_col] = df[txt_col].fillna("").astype(str)
    df["__txt"] = df[txt_col].map(normalize_text)
    df = df[df["__txt"].str.len() >= MIN_TEXT_LEN]
    if df.empty: raise RuntimeError("No products left after filtering.")

    n = len(df)
    shard_rows = args.shard_rows
    n_shards = math.ceil(n / shard_rows)
    log(f"[product] rows={n:,}  shards={n_shards}x{shard_rows}  batch={args.batch}")

    model = get_model(args.model, args.device)

    vecs_list = []
    for si in range(n_shards):
        a = si * shard_rows
        b = min((si+1)*shard_rows, n)
        chunk = df.iloc[a:b]
        texts = chunk["__txt"].tolist()
        emb = encode_shard(model, texts, args.batch)
        vecs_list.append(emb)
        pct = (b / n) * 100
        log(f"[product] shard {si+1}/{n_shards}  rows {a}-{b-1}  dim={emb.shape[1]}  ({pct:.1f}%)")

    V = np.vstack(vecs_list)
    assert V.shape[0] == n
    PROD_OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(PROD_OUT_EMB, V)
    meta = df[["sku", "n_reviews", "avg_stars", "last_ts", txt_col]].copy()
    for c in ["n_reviews","avg_stars","last_ts"]:
        if c not in meta.columns: meta[c] = np.nan
    meta = meta.rename(columns={txt_col:"agg_text"})
    meta.to_parquet(PROD_OUT_META, index=False)
    log(f"[ok] wrote {PROD_OUT_EMB} shape={V.shape}")
    log(f"[ok] wrote {PROD_OUT_META} rows={len(meta):,}")

# ---------------- REVIEWS ----------------
def build_review_embeddings(args):
    src = Path(args.input or REV_IN)
    if not src.exists():
        raise FileNotFoundError(f"Review input not found: {src}")
    need = {"id","sku","text"}
    df = pd.read_parquet(src)
    miss = need - set(df.columns)
    if miss: raise ValueError(f"{src} missing {sorted(miss)}")
    df = df[["id","sku","ts","stars","text"]].copy()
    df["id"] = df["id"].astype(str)
    df["sku"] = df["sku"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    if "stars" in df.columns: df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    if "ts" in df.columns: df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # clean + spam + dedup
    df["__txt"] = df["text"].map(normalize_text)
    df = df[df["__txt"].str.len() >= MIN_TEXT_LEN]
    if args.no_spam is False:
        before = len(df); df = df[~df["__txt"].apply(looks_spammy)]
        log(f"[review] spam filtered {before-len(df):,} rows")
    if args.no_dedup is False:
        before = len(df); df = df.drop_duplicates(subset=["sku","__txt"])
        log(f"[review] dedup removed {before-len(df):,}")

    n = len(df)
    if n == 0: raise RuntimeError("No reviews left after filtering.")
    shard_rows = args.shard_rows
    n_shards = math.ceil(n / shard_rows)
    log(f"[review] rows={n:,}  shards={n_shards}x{shard_rows}  batch={args.batch}")

    # prepare writer (append mode for resume)
    REV_OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    written = 0
    if Path(REV_OUT).exists() and args.resume:
        # count existing rows to resume
        try:
            existing = pq.read_table(REV_OUT, columns=["id"]).num_rows
            written = existing
            log(f"[resume] found existing {existing:,} rows in {REV_OUT}; appending remaining.")
        except Exception:
            pass

    model = get_model(args.model, args.device)

    # process shards
    for si in range(n_shards):
        a = si * shard_rows
        b = min((si+1)*shard_rows, n)
        if a < written:
            # already done in resume mode
            log(f"[review] skipping shard {si+1}/{n_shards} (rows {a}-{b-1}) — already written")
            continue
        chunk = df.iloc[a:b].copy()
        texts = chunk["__txt"].tolist()
        emb = encode_shard(model, texts, args.batch)

        # build arrow table for append
        chunk = chunk.drop(columns=["__txt"])
        emb_list = emb.tolist()
        tbl = pa.Table.from_pandas(
            chunk.assign(embedding=emb_list),
            preserve_index=False
        )
        if writer is None:
            writer = pq.ParquetWriter(REV_OUT, tbl.schema)
        writer.write_table(tbl)
        written = b
        pct = (written / n) * 100
        log(f"[review] shard {si+1}/{n_shards}  rows {a}-{b-1}  ({pct:.1f}%)")

    if writer is not None:
        writer.close()
    log(f"[ok] wrote {REV_OUT} total rows={n:,}")

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Build embeddings for products or reviews (sharded, progress, resume).")
    ap.add_argument("--target", choices=["product","review"], required=True)
    ap.add_argument("--input", type=str, default="")
    ap.add_argument("--text-col", type=str, default="")
    ap.add_argument("--model", type=str, default=EMB_MODEL_DEFAULT)
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 … (default: auto)")
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT)
    ap.add_argument("--shard-rows", type=int, default=20000, help="rows per shard to encode")
    # review-only options
    ap.add_argument("--resume", action="store_true", help="append to existing reviews parquet")
    ap.add_argument("--no-spam", action="store_true", help="disable spam filter")
    ap.add_argument("--no-dedup", action="store_true", help="disable (sku,text) dedup")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    log(f"[args] {args}")
    if args.target == "product":
        build_product_embeddings(args)
    else:
        build_review_embeddings(args)
