# nlp/10_product_prep.py
# Build product rows (one per SKU) directly from reviews_merged.parquet.
# Output: data/processed/products.parquet with columns:
#   sku, n_reviews, avg_stars, last_ts, agg_text
#
# Usage:
#   python nlp/10_product_prep.py \
#       --in  data/processed/reviews_merged.parquet \
#       --out data/processed/products.parquet \
#       --max-reviews-per-sku 80

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

DEF_IN  = Path("data/processed/reviews_merged.parquet")
DEF_OUT = Path("data/processed/products.parquet")

def normalize_text(s: str) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    s = " ".join(s.split())
    return s

def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    cols = ["id","sku","ts","stars","text"]
    df = pd.read_parquet(path, columns=[c for c in cols if c in pd.read_parquet(path).columns])
    need = {"id","sku","text"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)} (need at least id, sku, text)")
    df["id"] = df["id"].astype(str)
    df["sku"] = df["sku"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    if "stars" in df.columns:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # drop empty/very short texts
    df = df[df["text"].str.len() >= 10].copy()
    return df

def build_products(df: pd.DataFrame, max_reviews_per_sku: int = 80) -> pd.DataFrame:
    # clean text + lightweight dedup by (sku, normalized text)
    df["__clean"] = df["text"].map(normalize_text)
    before = len(df)
    df = df[df["__clean"].str.len() >= 10]
    df = df.drop_duplicates(subset=["sku","__clean"])
    deduped = before - len(df)

    # per-SKU KPIs
    grp = df.groupby("sku", as_index=False)
    kpis = grp.agg(
        n_reviews=("id","count"),
        avg_stars=("stars","mean"),
        last_ts=("ts","max")
    )

    # choose up to N texts per SKU (prefer higher stars, then recency)
    if "stars" not in df.columns:
        df["stars"] = np.nan
    if "ts" not in df.columns:
        df["ts"] = pd.NaT

    df = df.sort_values(["sku","stars","ts"], ascending=[True, False, False])
    # rank within sku and keep top N
    df["__rank"] = df.groupby("sku").cumcount() + 1
    df_keep = df[df["__rank"] <= max_reviews_per_sku].copy()

    # aggregate texts
    agg_txt = (df_keep
               .groupby("sku")["__clean"]
               .apply(lambda ss: " \n".join(ss.tolist()))
               .rename("agg_text")
               .reset_index())

    products = kpis.merge(agg_txt, on="sku", how="left")
    # tidy types
    products["avg_stars"] = products["avg_stars"].astype(float).round(3)
    # ensure strings
    products["agg_text"] = products["agg_text"].fillna("")
    return products, deduped

def main():
    ap = argparse.ArgumentParser(description="Create products.parquet from reviews_merged.parquet (no topics needed).")
    ap.add_argument("--in",  dest="inp",  default=str(DEF_IN),  help="Input reviews parquet (default: data/processed/reviews_merged.parquet)")
    ap.add_argument("--out", dest="out", default=str(DEF_OUT), help="Output products parquet (default: data/processed/products.parquet)")
    ap.add_argument("--max-reviews-per-sku", type=int, default=80, help="Max reviews to concatenate per SKU (default: 80)")
    args = ap.parse_args()

    inp  = Path(args.inp)
    outp = Path(args.out)

    df = load_reviews(inp)
    products, deduped = build_products(df, max_reviews_per_sku=args.max_reviews_per_sku)

    outp.parent.mkdir(parents=True, exist_ok=True)
    products.to_parquet(outp, index=False)

    print(f"[OK] products.parquet written: {outp}")
    print(f"     products: {len(products):,} | deduped review rows: {deduped:,}")
    print("     sample:")
    cols = ["sku","n_reviews","avg_stars","last_ts","agg_text"]
    print(products[cols].head(3).to_string(index=False))

if __name__ == "__main__":
    main()
