#!/usr/bin/env python3
import os, sys, hashlib
from datetime import datetime
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)

# ---------- Helpers ----------
def stable_id(text, ts, sku):
    key = f"{text or ''}|{ts or ''}|{sku or ''}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]

def clean_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    # Canonical columns
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # Map common variants
    TEXT_COLS  = ["text","reviewtext","content","body","review_body","review_text"]
    STAR_COLS  = ["stars","rating","overall","score","star_rating"]
    DATE_COLS  = ["date","reviewtime","created_at","timestamp","unixreviewtime"]
    SKU_COLS   = ["sku","asin","product_id","item_id","productid"]
    ID_COLS    = ["id","review_id","reviewerid"]

    def pick(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        return None

    c_text = pick(TEXT_COLS)
    c_star = pick(STAR_COLS)
    c_date = pick(DATE_COLS)
    c_sku  = pick(SKU_COLS)
    c_id   = pick(ID_COLS)

    if c_text is None or c_star is None:
        raise ValueError(f"[{source_name}] Missing required columns. Have: {list(df.columns)}")

    out = pd.DataFrame()
    out["text"] = df[c_text].astype(str).str.strip()

    # Stars to int 1..5
    stars = pd.to_numeric(df[c_star], errors="coerce")
    # some sources store as floats; round to nearest int in [1..5]
    stars = stars.round().astype("Int64")
    out["stars"] = stars
    out = out[out["stars"].between(1,5, inclusive="both")]

    # SKU
    out["sku"] = df[c_sku].astype(str) if c_sku else None

    # Date → ISO UTC
    ts = None
    if c_date:
        if "unix" in c_date:  # unixReviewTime
            ts = pd.to_datetime(df[c_date], unit="s", errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(df[c_date], errors="coerce", utc=True)
    out["ts"] = ts

    # Source tag
    out["source"] = source_name

    # ID: prefer existing else hash(text|ts|sku)
    if c_id and c_id in df.columns:
        ids = df[c_id].astype(str)
    else:
        ids = pd.Series([None]*len(out))
    out["id"] = [
        (i if (i and i.strip()) else stable_id(t, (ts[i_idx].isoformat() if (ts is not None and pd.notna(ts.iloc[i_idx])) else None), (out["sku"].iloc[i_idx] if "sku" in out.columns else None)))
        for i_idx, i in enumerate(ids)
    ]

    # Basic text quality
    out = out[out["text"].str.len() >= 10]
    out = out.drop_duplicates(subset=["id"])

    # Column order
    out = out[["id","sku","ts","stars","text","source"]]
    return out

# ---------- Process Kaggle CSV ----------
def process_kaggle():
    # adjust path if your CSV differs
    candidates = [
        os.path.join(RAW_DIR, "kaggle_reviews", "amazon_product_reviews.csv",),
        os.path.join(RAW_DIR, "amazon_product_reviews.csv"),
        os.path.join(RAW_DIR, "kaggle_reviews", "Reviews.csv"),
    ]
    print(f"[Candidates]: {candidates}")
    kaggle_path = next((p for p in candidates if os.path.exists(p)), None)
    if kaggle_path is None:
        print("WARN: Kaggle CSV not found; skipping Kaggle source.", file=sys.stderr)
        return pd.DataFrame(columns=["id","sku","ts","stars","text","source"])

    print(f"[Kaggle] Loading {kaggle_path}")
    df = pd.read_csv(kaggle_path, low_memory=False)
    out = clean_df(df, "kaggle")
    out.to_parquet(os.path.join(PROC_DIR, "kaggle_norm.parquet"), index=False)
    print(f"[Kaggle] Normalized rows: {len(out)}")
    return out

# ---------- Process SNAP JSON (large; stream in chunks) ----------
def process_snap_json():
    json_path = os.path.join(RAW_DIR, "reviews_Electronics_5.json")
    if not os.path.exists(json_path):
        print("WARN: SNAP JSON not found; skipping SNAP source.", file=sys.stderr)
        return pd.DataFrame(columns=["id","sku","ts","stars","text","source"])

    print(f"[SNAP] Streaming {json_path}")
    # Read in chunks to avoid OOM
    chunks = []
    reader = pd.read_json(json_path, lines=True, chunksize=100_000)
    for chunk in tqdm(reader, desc="[SNAP] chunks"):
        # SNAP fields: reviewText, overall, asin, unixReviewTime/reviewTime
        # Keep only necessary columns to reduce mem
        keep_cols = [c for c in chunk.columns if c.lower() in {
            "reviewtext","overall","asin","reviewtime","unixreviewtime","reviewerid","rating","stars","text","date","product_id","item_id"
        }]
        chunk = chunk[keep_cols]
        cleaned = clean_df(chunk, "snap_electronics")
        chunks.append(cleaned)

    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["id","sku","ts","stars","text","source"])
    out.to_parquet(os.path.join(PROC_DIR, "snap_norm.parquet"), index=False)
    print(f"[SNAP] Normalized rows: {len(out)}")
    return out

def main():
    kaggle_df = process_kaggle()
    snap_df = process_snap_json()

    merged = pd.concat([kaggle_df, snap_df], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["id"])
    print(f"[Merge] total={before} after_dedup={len(merged)}")

    # Save full merged
    merged_path_parquet = os.path.join(PROC_DIR, "reviews_merged.parquet")
    merged.to_parquet(merged_path_parquet, index=False)

    # Also a CSV (smaller sample for quick iteration)
    sample_n = min(100_000, len(merged))
    if sample_n > 0:
        sample = merged.sample(n=sample_n, random_state=42)
        sample_csv = os.path.join(PROC_DIR, "reviews_merged_sample_100k.csv")
        sample.to_csv(sample_csv, index=False)
        print(f"[Merge] Wrote sample CSV: {sample_csv} ({sample_n} rows)")

    print(f"[OK] Wrote merged Parquet: {merged_path_parquet} ({len(merged)} rows)")

if __name__ == "__main__":
    main()
