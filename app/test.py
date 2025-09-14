#!/usr/bin/env python3
# app/test.py
# General product search over your artifacts.
# Inputs (required): data/processed/product_emb.npy, product_emb_meta.parquet
# Optional:          data/processed/product_bm25.pkl (from nlp/12_product_prep.py)
#                    data/processed/reviews_with_embeddings.parquet (for best-review snippet)
#
# Example:
#   python app/test.py -q "best budget wireless keyboard for programming" -k 10
#   python app/test.py -q "noise cancelling headphones for flights" -k 10 --rerank_k 50 --w-bm25 0.2 --w-rerank 0.2

from __future__ import annotations
import argparse, json, math, os, pickle, sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# ----------- Paths -----------
DATA = Path("data/processed")
P_EMB = DATA / "product_emb.npy"
P_META= DATA / "product_emb_meta.parquet"
REV_EMB = DATA / "reviews_with_embeddings.parquet"     # optional
BM25_PKL= DATA / "product_bm25.pkl"                    # optional

# ----------- Models -----------
EMB_MODEL = os.environ.get("EMB_MODEL", "BAAI/bge-small-en-v1.5")   # must match your build
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Lazy imports
def _load_st_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL)

def _load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)

def _load_rankbm25():
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi
    except Exception:
        return None

# ----------- Utils -----------
def log(msg: str): print(msg, flush=True)

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0: return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo + 1e-12)).astype(np.float32)

def bayesian_prior(avg: np.ndarray, n: np.ndarray, C: float = 20.0, global_mean: float | None = None) -> np.ndarray:
    g = float(np.nanmean(avg)) if global_mean is None else float(global_mean)
    return ((avg * n) + (g * C)) / (n + C + 1e-9)

def cosine_search(qvec: np.ndarray, mat: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    # mat: (N, D) normalized; qvec: (D,)
    sims = mat @ qvec
    if topk >= len(sims): topk = len(sims)
    idx = np.argpartition(-sims, topk-1)[:topk]
    # sort topk
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def load_product_index() -> Tuple[pd.DataFrame, np.ndarray]:
    if not P_EMB.exists() or not P_META.exists():
        raise SystemExit("[ERR] product_emb.npy and/or product_emb_meta.parquet missing in data/processed/")
    meta = pd.read_parquet(P_META)
    if "sku" not in meta.columns or "agg_text" not in meta.columns:
        raise SystemExit("[ERR] product_emb_meta.parquet must have 'sku' and 'agg_text'")
    V = np.load(P_EMB, mmap_mode="r").astype(np.float32)
    if len(meta) != V.shape[0]:
        raise SystemExit(f"[ERR] length mismatch: meta={len(meta)} vs emb_rows={V.shape[0]}")
    # normalize once for cosine
    Vn = l2_normalize(np.array(V), axis=1)
    return meta.reset_index(drop=True), Vn

def load_bm25() -> Tuple[object, List[str]] | None:
    if not BM25_PKL.exists(): return None
    BM25Okapi = _load_rankbm25()
    if BM25Okapi is None:
        log("[warn] rank_bm25 not installed; skipping BM25. Install with: pip install rank_bm25")
        return None
    with open(BM25_PKL, "rb") as f:
        blob = pickle.load(f)
    corpus = blob["corpus"]
    bm25 = BM25Okapi(corpus)
    return bm25, blob["skus"]

def ensure_same_order(meta: pd.DataFrame, bm25_skus: List[str]) -> List[int] | None:
    # returns an index mapping to reorder bm25 docs to meta order, or None if mismatch
    idx_map = {s: i for i, s in enumerate(bm25_skus)}
    try:
        order = [idx_map[s] for s in meta["sku"].astype(str).tolist()]
        return order
    except KeyError:
        return None

def bm25_scores(bm25, query_tokens: List[str], order_idx: List[int] | None, top_idx: np.ndarray) -> np.ndarray:
    # score only candidates (top_idx) for speed
    scores_all = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
    if order_idx is not None:
        scores_all = scores_all[np.array(order_idx)]
    return scores_all[top_idx]

def tokenize_query(q: str) -> List[str]:
    import re
    STOP = {"a","an","the","and","or","of","for","to","in","on","with","is","are","it","this","that"}
    toks = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", q.lower())
    return [t for t in toks if t not in STOP]

def best_review_snippets(qvec: np.ndarray, cand_skus: List[str], max_rows:int=1_000_000) -> Dict[str, Dict]:
    """Return per-SKU best review {'score':float, 'text':str, 'stars':float} using reviews_with_embeddings.parquet.
       If file missing, returns {}.
    """
    if not REV_EMB.exists(): return {}
    cols = ["sku","text","stars","embedding"]
    df = pd.read_parquet(REV_EMB, columns=[c for c in cols if c != "embedding"])
    # quick early exit if 'sku' missing
    if "sku" not in df.columns:
        log("[warn] reviews_with_embeddings.parquet lacks 'sku'; skipping snippets.")
        return {}
    # We’ll read embeddings separately to avoid loading all at once
    df["__sel"] = df["sku"].astype(str).isin(set(cand_skus))
    sub_meta = df[df["__sel"]].drop(columns="__sel")
    if sub_meta.empty: return {}

    # Load embeddings column only (object -> np.ndarray) for selected rows
    emb_series = pd.read_parquet(REV_EMB, columns=["embedding"]).iloc[sub_meta.index]
    # cap rows to keep memory predictable
    if len(sub_meta) > max_rows:
        sub_meta = sub_meta.iloc[:max_rows]
        emb_series = emb_series.iloc[:max_rows]

    E = np.stack(emb_series["embedding"].values).astype(np.float32)   # (M,D)
    En = l2_normalize(E, axis=1)
    sims = En @ qvec  # (M,)
    # pick best per SKU
    sub_meta = sub_meta.reset_index(drop=True)
    sub_meta["__sim"] = sims
    best = {}
    for sku, grp in sub_meta.groupby("sku"):
        j = int(grp["__sim"].values.argmax())
        row = grp.iloc[j]
        best[str(sku)] = {"score": float(row["__sim"]), "text": str(row["text"])[:400], "stars": float(row.get("stars", np.nan))}
    return best

def cross_encoder_scores(query: str, texts: List[str]) -> np.ndarray:
    try:
        ce = _load_cross_encoder()
    except Exception as e:
        log(f"[warn] cross-encoder load failed: {e}; skipping reranker.")
        return np.zeros(len(texts), dtype=np.float32)
    pairs = [(query, t) for t in texts]
    scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
    return np.array(scores, dtype=np.float32)

# ----------- Main search -----------
def search(args):
    meta, V = load_product_index()
    # encode query
    try:
        st = _load_st_encoder()
        qvec = st.encode([args.query], normalize_embeddings=True)[0].astype(np.float32)
    except Exception as e:
        raise SystemExit(f"[ERR] loading/encoding with {EMB_MODEL} failed: {e}")

    # dense retrieval to topK0 (pool for reranker & scoring)
    topK0 = max(args.k, args.rerank_k, 100)
    cand_idx, dense_scores = cosine_search(qvec, V, topK0)
    cand = meta.iloc[cand_idx].reset_index(drop=True).copy()
    cand["_dense"] = dense_scores.astype(np.float32)

    # BM25 (optional)
    bm25_pair = load_bm25()
    if bm25_pair:
        bm25, bm25_skus = bm25_pair
        order = ensure_same_order(meta, bm25_skus)
        qtoks = tokenize_query(args.query)
        cand["_bm25_raw"] = bm25_scores(bm25, qtoks, order, cand_idx)
        cand["_bm25"] = minmax(cand["_bm25_raw"].values)
    else:
        cand["_bm25"] = 0.0

    # Priors (Bayesian avg on stars + log volume)
    n = pd.to_numeric(cand.get("n_reviews", pd.Series([np.nan]*len(cand))), errors="coerce").fillna(0).values
    r = pd.to_numeric(cand.get("avg_stars", pd.Series([np.nan]*len(cand))), errors="coerce").fillna(np.nan).values
    prior_rating = bayesian_prior(r, n, C=args.prior_C)
    prior_volume = np.log1p(n) / (np.log1p(n).max() + 1e-9)
    cand["_prior"] = minmax(prior_rating) * 0.7 + 0.3 * prior_volume

    # Cross-encoder reranker over top rerank_k
    if args.rerank_k > 0:
        k_rr = min(args.rerank_k, len(cand))
        rr_texts = cand["agg_text"].astype(str).str.slice(0, 2000).tolist()[:k_rr]
        rr_scores = cross_encoder_scores(args.query, rr_texts)
        rr_norm = minmax(rr_scores)
        zeros = np.zeros(len(cand), dtype=np.float32)
        zeros[:k_rr] = rr_norm
        cand["_rerank"] = zeros
    else:
        cand["_rerank"] = 0.0

    # Best review snippet per SKU (optional heavy)
    snippets = {}
    if not args.no_snippets and REV_EMB.exists():
        skus = cand["sku"].astype(str).tolist()
        snippets = best_review_snippets(qvec, skus, max_rows=args.max_reviews_scan)

    # Normalize dense
    cand["_dense"] = minmax(cand["_dense"].values)

    # Optional best-review contribution
    best_contrib = np.zeros(len(cand), dtype=np.float32)
    if snippets:
        for i, s in enumerate(cand["sku"].astype(str).tolist()):
            v = snippets.get(s, {}).get("score")
            if v is not None: best_contrib[i] = v
        best_contrib = minmax(best_contrib)
    cand["_bestrev"] = best_contrib

    # Final score
    final = (
        args.w_dense  * cand["_dense"].values +
        args.w_bm25   * cand["_bm25"].values +
        args.w_rerank * cand["_rerank"].values +
        args.w_prior  * cand["_prior"].values +
        args.w_best   * cand["_bestrev"].values
    ).astype(np.float32)

    cand["_final"] = final
    cand = cand.sort_values("_final", ascending=False).head(args.k).reset_index(drop=True)

    # Attach snippet text
    out_rows = []
    for _, row in cand.iterrows():
        s = str(row["sku"])
        snip = snippets.get(s) if snippets else None
        out_rows.append({
            "sku": s,
            "score": round(float(row["_final"]), 4),
            "dense": round(float(row["_dense"]), 4),
            "bm25": round(float(row["_bm25"]), 4),
            "rerank": round(float(row["_rerank"]), 4),
            "prior": round(float(row["_prior"]), 4),
            "bestrev": round(float(row["_bestrev"]), 4),
            "n_reviews": int(row.get("n_reviews", 0) if pd.notna(row.get("n_reviews", np.nan)) else 0),
            "avg_stars": round(float(row.get("avg_stars", np.nan)), 2) if pd.notna(row.get("avg_stars", np.nan)) else None,
            "snippet_stars": float(snip["stars"]) if snip and snip.get("stars") is not None else None,
            "snippet": snip["text"] if snip else None,
        })

    # Print
    print("\nTop results:")
    for i, r in enumerate(out_rows, 1):
        print(f"[{i}] {r['sku']}  score={r['score']}  (dense={r['dense']} bm25={r['bm25']} rerank={r['rerank']} prior={r['prior']} best={r['bestrev']})  "
              f"reviews={r['n_reviews']} avg={r['avg_stars']}")
        if r["snippet"]:
            print("    ", r["snippet"])

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({"query": args.query, "results": out_rows}, f, ensure_ascii=False, indent=2)
        print(f"\n[ok] wrote {args.json_out}")

# ----------- CLI -----------
def parse_args():
    ap = argparse.ArgumentParser(description="Search products with dense + (optional) BM25 + reranker + priors + snippets")
    ap.add_argument("-q","--query", required=True, help="User query")
    ap.add_argument("-k","--k", type=int, default=10, help="How many results to show")
    ap.add_argument("--rerank_k", type=int, default=50, help="Cross-encoder rerank pool size (0 to disable)")
    ap.add_argument("--no-snippets", action="store_true", help="Disable best-review snippet scoring")
    ap.add_argument("--max-reviews-scan", type=int, default=1_000_000, help="Cap on reviews scanned for snippets")
    # weights
    ap.add_argument("--w-dense", type=float, default=0.55)
    ap.add_argument("--w-bm25", type=float, default=0.15)
    ap.add_argument("--w-rerank", type=float, default=0.15)
    ap.add_argument("--w-prior", type=float, default=0.10)
    ap.add_argument("--w-best", type=float, default=0.05)
    ap.add_argument("--prior-C", type=float, default=20.0, help="Bayesian prior strength")
    ap.add_argument("--json-out", type=str, default="", help="Optional path to save results JSON")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    search(args)
