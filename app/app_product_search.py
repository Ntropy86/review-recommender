# app/streamlit_app.py  — polished UI + gates + small-sample damping
from __future__ import annotations
import math, json, pickle, time, re, logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config

# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)

# ---------- Paths (from config) ----------
P_EMB   = config.PRODUCT_EMB_PATH
P_META  = config.PRODUCT_META_PATH
REV_EMB = config.REVIEWS_EMB_PATH
BM25_PKL= config.BM25_PATH

EMB_MODEL    = config.EMB_MODEL
RERANK_MODEL = config.RERANK_MODEL

# ---------- Look & Feel ----------
st.set_page_config(page_title=config.APP_TITLE, page_icon="✨", layout="wide")

# Simple health check endpoint
if len(st.query_params) > 0 and "health" in st.query_params:
    st.write("OK")
    st.stop()

st.markdown("""
<style>
:root { --fg:#111827; --soft:#6b7280; --bg:#0b1020; --card:#111827; --chip:#0ea5a4; }
.block-container { padding-top:1rem; }
.stApp { background: linear-gradient(180deg, #0b1020, #0d1326); color:#e5e7eb; }
h1,h2,h3 { color:#f9fafb !important; }
.card { border:1px solid #1f2937; border-radius:16px; padding:14px 16px; margin-bottom:12px; background:#111827; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#0b3b49; color:#a7f3d0; margin-right:6px; font-size:12px;}
.metric { background:#0f172a; border:1px solid #1f2937; border-radius:12px; padding:6px 10px; margin-right:8px; display:inline-block;}
.small { color:#9ca3af; font-size:0.85rem;}
.stSlider label, .stCheckbox label, .stNumberInput label { color:#e5e7eb !important;}
.stButton>button { border-radius:12px; background:#22d3ee; color:#0b1020; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ---------- Caches ----------
@st.cache_resource(show_spinner=False)
def _st_encoder(name: str):
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence transformer model: {name}")
        model = SentenceTransformer(name)
        logger.info(f"Successfully loaded sentence transformer: {name}")
        return model
    except ImportError as e:
        logger.error(f"SentenceTransformers library not available: {e}")
        st.error("❌ SentenceTransformers library not installed. Please install with: `pip install sentence-transformers`")
        st.stop()
    except Exception as e:
        logger.error(f"Failed to load sentence transformer model {name}: {e}")
        st.error(f"❌ Failed to load embedding model `{name}`: {str(e)}")
        st.info("💡 This might be due to network issues or an invalid model name. Try again or check your internet connection.")
        st.stop()

@st.cache_resource(show_spinner=False)
def _cross_encoder(name: str):
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder model: {name}")
        model = CrossEncoder(name)
        logger.info(f"Successfully loaded cross-encoder: {name}")
        return model
    except ImportError as e:
        logger.warning(f"SentenceTransformers library not available for cross-encoder: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load cross-encoder model {name}: {e}")
        if config.ENABLE_RERANKING:
            st.warning(f"⚠️ Cross-encoder reranking disabled: failed to load model `{name}` ({str(e)})")
        return None

@st.cache_data(show_spinner=False)
def _product_index() -> Tuple[pd.DataFrame, np.ndarray]:
    try:
        if not P_EMB.exists():
            logger.error(f"Product embeddings file not found: {P_EMB}")
            st.error(f"❌ Product embeddings file missing: `{P_EMB.name}`")
            st.info("💡 Please run the data preprocessing pipeline first:\n```bash\npython nlp/10_product_prep.py\npython nlp/11_build_product_embeddings.py\n```")
            st.stop()
        
        if not P_META.exists():
            logger.error(f"Product metadata file not found: {P_META}")
            st.error(f"❌ Product metadata file missing: `{P_META.name}`")
            st.info("💡 Please run the data preprocessing pipeline first:\n```bash\npython nlp/10_product_prep.py\n```")
            st.stop()
        
        logger.info(f"Loading product metadata from {P_META}")
        meta = pd.read_parquet(P_META)
        
        logger.info(f"Loading product embeddings from {P_EMB}")
        V = np.load(P_EMB, mmap_mode="r").astype(np.float32)
        
        if len(meta) != V.shape[0]:
            logger.error(f"Dimension mismatch: meta rows ({len(meta)}) != embedding rows ({V.shape[0]})")
            st.error(f"❌ Data inconsistency: metadata has {len(meta)} rows but embeddings have {V.shape[0]} rows")
            st.info("💡 Please rebuild the embeddings to match the metadata.")
            st.stop()
        
        logger.info(f"Successfully loaded {len(meta)} products with {V.shape[1]}-dimensional embeddings")
        Vn = _l2norm(np.array(V), axis=1)
        return meta.reset_index(drop=True), Vn
        
    except Exception as e:
        logger.error(f"Failed to load product index: {e}")
        st.error(f"❌ Failed to load product data: {str(e)}")
        st.info("💡 Please check that your data files are valid and not corrupted.")
        st.stop()

@st.cache_resource(show_spinner=False)
def _bm25_loader():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        logger.warning(f"BM25 library not available: {e}")
        if config.ENABLE_BM25:
            st.warning("⚠️ BM25 search disabled: `rank_bm25` library not installed. Install with: `pip install rank-bm25`")
        return None
    except Exception as e:
        logger.error(f"Failed to import BM25: {e}")
        return None
    
    if not BM25_PKL.exists():
        logger.info(f"BM25 index file not found: {BM25_PKL}")
        if config.ENABLE_BM25:
            st.info(f"ℹ️ BM25 search disabled: index file `{BM25_PKL.name}` not found. Run preprocessing to create it.")
        return None
    
    try:
        logger.info(f"Loading BM25 index from {BM25_PKL}")
        with open(BM25_PKL, "rb") as f:
            blob = pickle.load(f)
        
        bm25_index = BM25Okapi(blob["corpus"])
        logger.info(f"Successfully loaded BM25 index with {len(blob['skus'])} documents")
        return {"bm25": bm25_index, "skus": [str(s) for s in blob["skus"]]}
        
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        st.warning(f"⚠️ BM25 search disabled: failed to load index file ({str(e)})")
        return None

# ---------- Utils ----------
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOP = {"a","an","the","and","or","of","for","to","in","on","with","is","are","it","this","that"}

SYN = {
    "sock": {"sock","socks"},
    "headphone": {"headphone","headphones","earphone","earphones","earbud","earbuds","headset"},
    "keyboard": {"keyboard","keyboards"},
    "wireless": {"wireless","bluetooth"},
    "noise": {"noise cancelling","noise-canceling","noise canceling","anc"},
    "cat": {"cat","cats","kitten","kittens","kitty"},
    "dog": {"dog","dogs","puppy","puppies"},
    "design": {"design","pattern","print","graphic","artwork","motif","theme"},
}
COLORS = {
    "yellow":{"yellow","mustard","lemon","gold","golden"},
    "red":{"red","scarlet","crimson","maroon"},
    "blue":{"blue","navy","cobalt","azure"},
    "green":{"green","emerald","olive"},
    "black":{"black"},
    "white":{"white","ivory"},
    "pink":{"pink","rose"},
    "purple":{"purple","violet","lavender"},
    "orange":{"orange","amber"},
    "brown":{"brown","tan","beige","khaki"},
    "gray":{"gray","grey","charcoal","slate"},
}

def _l2norm(x: np.ndarray, axis:int=1, eps:float=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True); n = np.maximum(n, eps); return x/n

def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x.astype(np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + 1e-12)).astype(np.float32)

def _tokenize(q: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(q.lower()) if t not in STOP]

def _cosine_pool(qvec: np.ndarray, mat: np.ndarray, pool: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = mat @ qvec; pool = min(pool, len(sims))
    idx = np.argpartition(-sims, pool-1)[:pool]; idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def _bayes_prior(avg: np.ndarray, n: np.ndarray, C: float = 20.0, gmean: float | None = None) -> np.ndarray:
    g = float(np.nanmean(avg)) if gmean is None else float(gmean)
    return ((avg * n) + (g * C)) / (n + C + 1e-9)

def _bm25_for_candidates(bm25_blob, query: str, cand_skus: List[str]) -> np.ndarray:
    if not bm25_blob: return np.zeros(len(cand_skus), dtype=np.float32)
    toks = _tokenize(query); 
    if not toks: return np.zeros(len(cand_skus), dtype=np.float32)
    bm25, skus = bm25_blob["bm25"], bm25_blob["skus"]
    scores_all = np.array(bm25.get_scores(toks), dtype=np.float32)
    by_sku = {skus[i]: scores_all[i] for i in range(len(skus))}
    return np.array([by_sku.get(str(s), 0.0) for s in cand_skus], dtype=np.float32)

# ---- Query -> gating groups (category/color/keywords) ----
def _build_gate_groups(query: str) -> List[set[str]]:
    ql = query.lower()
    groups: List[set[str]] = []
    # colors explicitly mentioned
    for cname, syns in COLORS.items():
        if any(w in ql for w in syns):
            groups.append(syns)
    # category / key nouns
    toks = _tokenize(query)
    for t in toks:
        if t in SYN: groups.append(SYN[t])
        elif len(t) >= 4: groups.append({t})
    # deduplicate identical sets
    uniq = []
    for g in groups:
        if g not in uniq: uniq.append(g)
    return uniq[:6]  # cap
def _gate_factor(text: str, groups: List[set[str]], penalty: float = 0.5) -> tuple[float,int,int]:
    tl = text.lower()
    hits = 0
    factor = 1.0
    for g in groups:
        ok = any(s in tl for s in g)
        if ok: hits += 1
        else: factor *= penalty
    return factor, hits, len(groups)

def _trust_from_reviews(n: np.ndarray, min_reviews:int, sat:int=50) -> np.ndarray:
    # linear ramp to min_reviews, then log saturation
    ramp = np.clip(n / max(min_reviews,1), 0, 1)
    satv = np.minimum(1.0, np.log1p(n) / np.log1p(max(sat,1)))
    return (0.6*ramp + 0.4*satv).astype(np.float32)

# ---------- Core search ----------
def run_search(query: str, k: int, rerank_k: int,
               w_dense: float, w_bm25: float, w_rerank: float, w_prior: float, w_best: float,
               prior_C: float, use_snips: bool, max_scan: int,
               min_reviews: int, gate_penalty: float) -> tuple[pd.DataFrame,dict,dict]:
    meta, V = _product_index()
    st_model = _st_encoder(EMB_MODEL)
    qvec = st_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    pool = max(k, rerank_k, 150)
    cand_idx, dense_scores = _cosine_pool(qvec, V, pool)
    cand = meta.iloc[cand_idx].reset_index(drop=True).copy()
    cand["_dense"] = _minmax(dense_scores.astype(np.float32))

    # BM25
    bm25_blob = _bm25_loader()
    bm25_raw = _bm25_for_candidates(bm25_blob, query, cand["sku"].astype(str).tolist())
    cand["_bm25"] = _minmax(bm25_raw)

    # Priors
    n = pd.to_numeric(cand.get("n_reviews", pd.Series([np.nan]*len(cand))), errors="coerce").fillna(0).values
    r = pd.to_numeric(cand.get("avg_stars", pd.Series([np.nan]*len(cand))), errors="coerce").fillna(np.nan).values
    prior_rating = _bayes_prior(r, n, C=prior_C)
    prior_volume = np.log1p(n) / (np.log1p(n).max() + 1e-9)
    cand["_prior"] = _minmax(prior_rating) * 0.7 + 0.3 * prior_volume

    # Reranker
    if rerank_k > 0:
        rr_k = min(rerank_k, len(cand))
        rr_texts = cand["agg_text"].astype(str).str.slice(0, 2000).tolist()[:rr_k]
        from_scores = _cross_encoder(RERANK_MODEL)
        if from_scores is None: rr = np.zeros(rr_k, dtype=np.float32)
        else:
            pairs = [(query, t) for t in rr_texts]
            rr = np.array(from_scores.predict(pairs, batch_size=64, show_progress_bar=False), dtype=np.float32)
        z = np.zeros(len(cand), dtype=np.float32); z[:rr_k] = _minmax(rr)
        cand["_rerank"] = z
    else:
        cand["_rerank"] = 0.0

    # Snippets (optional)
    snips = {}
    if use_snips and REV_EMB.exists():
        snips = _best_snippets(qvec, cand["sku"].astype(str).tolist(), max_rows=max_scan)
    best_contrib = np.zeros(len(cand), dtype=np.float32)
    if snips:
        for i, s in enumerate(cand["sku"].astype(str).tolist()):
            v = snips.get(s, {}).get("score")
            if v is not None: best_contrib[i] = v
        best_contrib = _minmax(best_contrib)
    cand["_best"] = best_contrib

    # Gates (category/color/keywords) and trust
    groups = _build_gate_groups(query)
    gate_vals = []
    for txt in cand["agg_text"].astype(str).str.slice(0, 6000).tolist():
        gf, _, _ = _gate_factor(txt, groups, penalty=gate_penalty)
        gate_vals.append(gf)
    cand["_gate"] = np.array(gate_vals, dtype=np.float32)
    cand["_trust"] = _trust_from_reviews(n, min_reviews=min_reviews, sat=80)

    # Final blend (then multiply by trust * gate)
    final = ( w_dense*cand["_dense"].values + w_bm25*cand["_bm25"].values +
              w_rerank*cand["_rerank"].values + w_prior*cand["_prior"].values +
              w_best*cand["_best"].values ).astype(np.float32)
    final = final * cand["_trust"].values * cand["_gate"].values
    cand["_final"] = final

    return cand.sort_values("_final", ascending=False).head(k).reset_index(drop=True), snips, {
        "bm25_active": bm25_blob is not None,
        "tokens": _tokenize(query),
        "groups": [list(g) for g in groups],
        "pool": pool
    }

# ---------- Snippets helper ----------
def _best_snippets(qvec: np.ndarray, cand_skus: List[str], max_rows:int=300_000) -> Dict[str, Dict]:
    try:
        if not REV_EMB.exists():
            logger.info(f"Review embeddings file not found: {REV_EMB}")
            return {}
        
        logger.info(f"Loading review embeddings for snippet extraction")
        cols = ["sku","text","stars","embedding"]
        
        # First check if required columns exist
        try:
            available_cols = pd.read_parquet(REV_EMB, nrows=1).columns.tolist()
            missing_cols = [c for c in ["sku", "text", "embedding"] if c not in available_cols]
            if missing_cols:
                logger.warning(f"Review embeddings missing required columns: {missing_cols}")
                return {}
        except Exception as e:
            logger.warning(f"Failed to check review embeddings schema: {e}")
            return {}
        
        meta = pd.read_parquet(REV_EMB, columns=[c for c in cols if c != "embedding"])
        if "sku" not in meta.columns:
            logger.warning("Review embeddings missing 'sku' column")
            return {}
        
        sel = meta["sku"].astype(str).isin(set(cand_skus))
        sub_meta = meta[sel]
        if sub_meta.empty:
            logger.info("No matching reviews found for candidate SKUs")
            return {}
        
        emb_series = pd.read_parquet(REV_EMB, columns=["embedding"]).iloc[sub_meta.index]
        if len(sub_meta) > max_rows:
            logger.info(f"Limiting review processing to {max_rows} rows")
            sub_meta = sub_meta.iloc[:max_rows]
            emb_series = emb_series.iloc[:max_rows]
        
        E = np.stack(emb_series["embedding"].values).astype(np.float32)
        En = _l2norm(E, axis=1)
        sims = En @ qvec
        sub_meta = sub_meta.reset_index(drop=True)
        sub_meta["__sim"] = sims
        
        best = {}
        for sku, grp in sub_meta.groupby("sku"):
            j = int(grp["__sim"].values.argmax())
            row = grp.iloc[j]
            best[str(sku)] = {
                "score": float(row["__sim"]), 
                "text": str(row["text"])[:600], 
                "stars": float(row.get("stars", np.nan))
            }
        
        logger.info(f"Found best snippets for {len(best)} products")
        return best
        
    except Exception as e:
        logger.error(f"Failed to extract review snippets: {e}")
        if config.ENABLE_SNIPPETS:
            st.warning(f"⚠️ Review snippets disabled due to error: {str(e)}")
        return {}

# ---------- UI ----------
tab_search, tab_metrics, tab_how = st.tabs(["🔎 Search", "📈 Metrics", "ℹ️ How it works"])

with tab_search:
    st.header("✨ Review Search Copilot")
    q = st.text_input("What are you looking for?", "best socks with kittens that are yellow")
    c1, c2, c3 = st.columns([1.2,1,1])
    k = c1.slider("Results (k)", 5, 25, config.DEFAULT_K, 1)
    rerank_k = c2.slider("Rerank pool", 0, 200, config.DEFAULT_RERANK_K, 10, help="Set 0 to disable cross-encoder.")
    min_reviews = c3.slider("Min reviews for full trust", 0, 50, config.DEFAULT_MIN_REVIEWS, 1)

    st.subheader("Weights")
    cc = st.columns(5)
    w_dense  = cc[0].slider("Dense",   0.0, 1.0, config.DEFAULT_W_DENSE, 0.05)
    w_bm25   = cc[1].slider("BM25",    0.0, 1.0, config.DEFAULT_W_BM25, 0.05)
    w_rerank = cc[2].slider("Rerank",  0.0, 1.0, config.DEFAULT_W_RERANK, 0.05)
    w_prior  = cc[3].slider("Prior",   0.0, 1.0, config.DEFAULT_W_PRIOR, 0.05)
    w_best   = cc[4].slider("Best review", 0.0, 1.0, config.DEFAULT_W_BEST, 0.05)

    c4, c5 = st.columns([1,1])
    gate_penalty = c4.slider("Penalty per missing attribute group", 0.1, 1.0, config.DEFAULT_GATE_PENALTY, 0.05,
                             help="Lower = stricter: missing color/category/keyword hurts more.")
    use_snips = c5.checkbox("Score & show best review snippet (heavy)", value=config.ENABLE_SNIPPETS)
    max_scan = st.select_slider("Max reviews scanned for snippets", options=[50_000, 100_000, 200_000, 300_000, 500_000], value=config.MAX_REVIEWS_SCAN)

    go = st.button("🚀 Search", type="primary", use_container_width=True)

    if go and q.strip():
        t0 = time.time()
        with st.spinner("Thinking…"):
            res, snips, dbg = run_search(q, k, rerank_k, w_dense, w_bm25, w_rerank, w_prior, w_best, 20.0, use_snips, max_scan, min_reviews, gate_penalty)
        t1 = time.time()
        st.caption(f"Done in **{(t1-t0):.2f}s** | Pool={dbg['pool']} | BM25={'✅' if dbg['bm25_active'] else '❌'} | Tokens: {', '.join(dbg['tokens']) or '—'} | Gates: {', '.join('/'.join(g) for g in dbg['groups']) or '—'}")

        for i, row in res.iterrows():
            sku = str(row["sku"])
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{i+1}. SKU:** `{sku}`  "
                        f"<span class='badge'>Reviews: {int(row.get('n_reviews',0))}</span> "
                        f"<span class='badge'>Avg ⭐ {float(row.get('avg_stars',np.nan)) if pd.notna(row.get('avg_stars',np.nan)) else '—'}</span>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<div class='metric'>Final: <b>{row['_final']:.3f}</b></div>"
                f"<div class='metric'>Dense: {row['_dense']:.3f}</div>"
                f"<div class='metric'>BM25: {row['_bm25']:.3f}</div>"
                f"<div class='metric'>Rerank: {row['_rerank']:.3f}</div>"
                f"<div class='metric'>Prior: {row['_prior']:.3f}</div>"
                f"<div class='metric'>BestRev: {row['_best']:.3f}</div>"
                f"<div class='metric'>Trust: {row['_trust']:.3f}</div>"
                f"<div class='metric'>Gate: {row['_gate']:.3f}</div>",
                unsafe_allow_html=True
            )
            if snips and sku in snips:
                s = snips[sku]
                with st.expander("Best review snippet", expanded=True):
                    st.write(s["text"])
                    if not math.isnan(s.get("stars", float("nan"))):
                        st.caption(f"Snippet ⭐ {s['stars']:.1f}")
            with st.expander("Product text (truncated)"):
                st.write(str(row["agg_text"])[:1500])
            st.markdown("</div>", unsafe_allow_html=True)

with tab_metrics:
    st.header("📈 Quick metrics (bring your own dev set)")
    st.caption("Upload JSONL lines: `{ \"query\": \"...\", \"relevant\": [\"SKU1\", \"SKU2\"] }`")
    up = st.file_uploader("Dev set (.jsonl)", type=["jsonl","json"], accept_multiple_files=False)
    eval_rerank_k = st.slider("Rerank pool (eval)", 0, 200, 50, 10)
    ew = st.columns(5)
    ew_dense  = ew[0].number_input("w_dense", 0.0, 1.0, 0.55, 0.05)
    ew_bm25   = ew[1].number_input("w_bm25", 0.0, 1.0, 0.20, 0.05)
    ew_rerank = ew[2].number_input("w_rerank", 0.0, 1.0, 0.20, 0.05)
    ew_prior  = ew[3].number_input("w_prior", 0.0, 1.0, 0.20, 0.05)
    ew_best   = ew[4].number_input("w_best",  0.0, 1.0, 0.10, 0.05)
    min_rev_eval = st.slider("Min reviews (trust) for eval", 0, 50, 8, 1)

    def dcg_at_k(rel: List[int], k: int) -> float:
        rel = rel[:k]; return sum((2**r - 1) / math.log2(i+2) for i,r in enumerate(rel))
    def ndcg_at_k(rank_skus: List[str], rel_skus: List[str], k: int = 10) -> float:
        rel_set = set(map(str, rel_skus)); rel = [1 if s in rel_set else 0 for s in rank_skus]
        ideal = sorted(rel, reverse=True); return 0.0 if sum(ideal)==0 else dcg_at_k(rel,k)/dcg_at_k(ideal,k)
    def mrr_at_k(rank_skus: List[str], rel_skus: List[str], k: int = 10) -> float:
        rel_set = set(map(str, rel_skus))
        for i,s in enumerate(rank_skus[:k],1):
            if s in rel_set: return 1.0/i
        return 0.0

    if up is not None:
        lines = [json.loads(l) for l in up.getvalue().decode("utf-8").splitlines() if l.strip()]
        meta, V = _product_index()
        st_model = _st_encoder(EMB_MODEL)
        rows = []
        for obj in lines:
            qq, rel = obj["query"], obj.get("relevant", [])
            res, _, _ = run_search(qq, 10, eval_rerank_k, ew_dense, ew_bm25, ew_rerank, ew_prior, ew_best, 20.0, False, 0, min_rev_eval, 0.5)
            rank = [str(s) for s in res["sku"].tolist()]
            rows.append({"query": qq, "nDCG@10": round(ndcg_at_k(rank, rel, 10),4),
                         "MRR@10": round(mrr_at_k(rank, rel, 10),4),
                         "first_hit": next((i+1 for i,s in enumerate(rank) if s in set(rel)), None),
                         "top1": rank[0] if rank else None})
        df = pd.DataFrame(rows)
        st.write(df)
        st.metric("Avg nDCG@10", f"{df['nDCG@10'].mean():.3f}")
        st.metric("Avg MRR@10",  f"{df['MRR@10'].mean():.3f}")

with tab_how:
    st.header("ℹ️ How it works (for recruiters)")
    st.markdown("""
**Pipeline (30 sec):**
1. Merge reviews → aggregate per SKU (*agg_text*, avg ⭐, count).
2. Encode products + reviews with `bge-small` (dense vectors).
3. Retrieval = cosine over product vectors (candidate pool).
4. **Keyword** BM25 adds exact-match signal.
5. **Cross-encoder** reranks the pool with a stronger relevance model.
6. **Priors** (Bayesian avg + volume) for robustness.
7. **Best review** per SKU via review vectors → explanation snippet.
8. **Guards**: small-sample penalty + attribute gates (color/category/keywords).

**Why it’s solid:** hybrid (semantic + keyword), reranked, statistically robust, and explainable with evidence.
""")
