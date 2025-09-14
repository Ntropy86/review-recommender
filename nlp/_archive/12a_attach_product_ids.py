#!/usr/bin/env python3
import os, sys, pandas as pd

PROC_REV = os.getenv("PROC_REV", "data/processed/reviews_with_topic_labels_llm.parquet")
RAW_PATH = os.getenv("RAW_PATH")  # e.g., data/raw/fine_food_reviews.parquet or .csv
PRODUCT_COL = os.getenv("PRODUCT_COL", "sku")  # your product key column in raw
OUT_REV = os.getenv("OUT_REV", PROC_REV)       # overwrite by default

def read_any(path):
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".csv"):     return pd.read_csv(path)
    if path.endswith(".tsv"):     return pd.read_csv(path, sep="\t")
    raise SystemExit(f"[ERROR] Unknown file type: {path}")

def main():
    if not RAW_PATH: raise SystemExit("[ERROR] Set RAW_PATH to your raw reviews file.")
    if not os.path.exists(PROC_REV): raise SystemExit(f"[ERROR] Missing {PROC_REV}")
    rev = pd.read_parquet(PROC_REV)
    if "id" not in rev.columns: raise SystemExit("[ERROR] Processed reviews missing 'id'; cannot join.")

    raw = read_any(RAW_PATH)
    if "id" not in raw.columns:
        if "Id" in raw.columns:
            raw = raw.rename(columns={"Id":"id"})
        else:
            raise SystemExit("[ERROR] Raw file has no 'id' column to join on.")

    if PRODUCT_COL not in raw.columns:
        raise SystemExit(f"[ERROR] PRODUCT_COL='{PRODUCT_COL}' not in raw. Found: {raw.columns.tolist()}")

    m = rev.merge(raw[["id", PRODUCT_COL]], on="id", how="left")
    m["sku"] = m[PRODUCT_COL].astype(str).str.strip()
    # keep existing 'source' for now, but downstream will use 'sku' via product prep
    print("[INFO] after join -> columns:", m.columns.tolist())
    print("[INFO] sku non-null rows:", m["sku"].notna().sum(), "distinct skus:", m["sku"].nunique())

    os.makedirs(os.path.dirname(OUT_REV) or ".", exist_ok=True)
    m.to_parquet(OUT_REV, index=False)
    print(f"[OK] wrote {OUT_REV}")

if __name__ == "__main__":
    main()
