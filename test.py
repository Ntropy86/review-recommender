# tools/audit_artifacts.py  (you can also name it test.py at repo root)
# Audits your data/processed artifacts for SKU and schema readiness.

import sys
from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

DATA = Path("data/processed")

# Expected files
REVIEWS              = DATA / "reviews_merged.parquet"          # must exist
P_META               = DATA / "product_emb_meta.parquet"        # must exist
P_EMB                = DATA / "product_emb.npy"                 # must exist
REVIEWS_EMB          = DATA / "reviews_with_embeddings.parquet" # optional but recommended
TOPIC_STATS          = DATA / "product_topic_stats.parquet"     # optional
TOPICS_NAMED_LLM     = DATA / "topics_named_llm.parquet"        # optional
TOPICS_NAMED         = DATA / "topics_named.parquet"            # optional
TOPIC_CARDS          = DATA / "topic_cards.parquet"             # optional
ASPECTS_FILE         = DATA / "topics_with_aspects_llm.parquet" # optional

# ---------- Safe serialization helpers ----------

def _to_py(obj):
    """Convert numpy/pandas scalars & datetimes/NaT to JSON-safe Python types."""
    if obj is None:
        return None
    # pandas NA/NaT
    if isinstance(obj, (pd._libs.missing.NAType,)):
        return None
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    # numpy datetime
    if isinstance(obj, (np.datetime64,)):
        try:
            return pd.to_datetime(obj).isoformat()
        except Exception:
            return str(obj)
    # numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # floats that are nan/inf
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

def df_sample_dict(df: pd.DataFrame, n: int = 5):
    """Return head(n) as a list of dicts with JSON-safe values."""
    if df is None or df.empty:
        return []
    head = df.head(n).copy()
    # Ensure object dtype so we can map safely
    head = head.astype("object")
    for col in head.columns:
        head[col] = head[col].map(_to_py)
    return head.to_dict(orient="records")

def safe_dump(obj, limit=600):
    """Dump any object to JSON safely, truncate for display."""
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return (s[:limit] + ("…" if len(s) > limit else ""))

# ---------- IO helpers ----------

def row_count(path: Path) -> int | None:
    try:
        if pq is not None:
            return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception:
        pass
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return None

def head_cols(path: Path, cols: list[str] | None = None, n: int = 3) -> pd.DataFrame | None:
    try:
        if cols is None:
            return pd.read_parquet(path).head(n)
        return pd.read_parquet(path, columns=cols).head(n)
    except Exception:
        return None

def exists(path: Path) -> bool:
    return path.exists()

def passfail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"

def coalesce_topics_named() -> pd.DataFrame | None:
    for p in [TOPICS_NAMED_LLM, TOPICS_NAMED]:
        if exists(p):
            try:
                df = pd.read_parquet(p)
                # normalize column names
                if "topic_id" not in df.columns:
                    for c in ["topic","cluster","label_id","topicid","topicId"]:
                        if c in df.columns:
                            df = df.rename(columns={c:"topic_id"})
                            break
                if "topic_label" not in df.columns:
                    for c in ["label","name","title","topic_name","card_title"]:
                        if c in df.columns:
                            df = df.rename(columns={c:"topic_label"})
                            break
                return df[["topic_id","topic_label"]]
            except Exception:
                continue
    return None

# ---------- Audit ----------

def audit():
    report = {"files": {}, "summary": {}, "notes": []}
    overall_ok = True

    # 1) reviews_merged
    f = str(REVIEWS)
    if not exists(REVIEWS):
        overall_ok = False
        report["files"][f] = {"exists": False, "message": "Missing required file."}
    else:
        required = ["id","sku","ts","stars","text"]
        df_head = head_cols(REVIEWS, None, 5)
        cols = list(df_head.columns) if df_head is not None else []
        # verify each required col can be read
        has = True
        for c in required:
            try:
                pd.read_parquet(REVIEWS, columns=[c])
            except Exception:
                has = False
                break
        null_sku = None
        try:
            df_small = pd.read_parquet(REVIEWS, columns=["sku"])
            null_sku = int(df_small["sku"].isna().sum())
        except Exception:
            pass
        ok = has and (null_sku is not None and null_sku < (row_count(REVIEWS) or 1))
        overall_ok &= ok
        report["files"][f] = {
            "exists": True,
            "row_count": row_count(REVIEWS),
            "required_cols_present": {c: True for c in required},
            "empty_sku_rows": null_sku,
            "status": passfail(ok),
            "sample": df_sample_dict(df_head, 5)
        }

    # 2) product embeddings
    fm = str(P_META)
    fe = str(P_EMB)
    pm_ok = exists(P_META)
    pe_ok = exists(P_EMB)
    if not pm_ok or not pe_ok:
        overall_ok = False
        report["files"][fm] = {"exists": pm_ok, "message": "Missing product meta."}
        report["files"][fe] = {"exists": pe_ok, "message": "Missing product embeddings."}
    else:
        try:
            m = pd.read_parquet(P_META)
            E = np.load(P_EMB)
            need = {"sku","n_reviews","avg_stars","agg_text"}
            ok_cols = need.issubset(set(m.columns))
            ok_len = len(m) == E.shape[0] and len(m) > 0
            ok_unique = m["sku"].isna().sum() == 0 and m["sku"].duplicated().sum() == 0
            ok = ok_cols and ok_len and ok_unique
            overall_ok &= ok
            report["files"][fm] = {
                "exists": True,
                "rows": len(m),
                "required_cols_present": {c: c in m.columns for c in need},
                "unique_sku": ok_unique,
                "status": passfail(ok),
                "sample": df_sample_dict(m[["sku","n_reviews","avg_stars"]], 5)
            }
            report["files"][fe] = {"exists": True, "shape": list(E.shape), "status": passfail(ok_len)}
        except Exception as e:
            overall_ok = False
            report["files"][fm] = {"exists": True, "error": repr(e), "trace": traceback.format_exc()}
            report["files"][fe] = {"exists": True}

    # 3) review embeddings (optional)
    fr = str(REVIEWS_EMB)
    if not exists(REVIEWS_EMB):
        report["files"][fr] = {"exists": False, "message": "Optional but recommended for best-review scoring."}
        rev_emb_ok = False
    else:
        try:
            rhead = head_cols(REVIEWS_EMB, None, 5)
            cols = set(rhead.columns) if rhead is not None else set(pd.read_parquet(REVIEWS_EMB).columns)
            need = {"id","text","embedding"}  # sku recommended
            has_need = need.issubset(cols)
            has_sku = "sku" in cols
            # test joinability via id->sku
            joinable = False
            if not has_sku and exists(REVIEWS):
                try:
                    left = pd.read_parquet(REVIEWS_EMB, columns=["id"]).head(10000)
                    right = pd.read_parquet(REVIEWS, columns=["id","sku"])
                    j = left.merge(right, on="id", how="left")
                    joinable = j["sku"].notna().mean() > 0.9
                except Exception:
                    joinable = False
            rev_emb_ok = has_need and (has_sku or joinable)
            report["files"][fr] = {
                "exists": True,
                "row_count": row_count(REVIEWS_EMB),
                "has_columns": {c: (c in cols) for c in ["id","sku","text","embedding","stars"]},
                "sku_present_or_joinable": bool(has_sku or joinable),
                "status": passfail(rev_emb_ok),
                "sample": df_sample_dict(rhead, 5)
            }
            if not has_sku and not joinable:
                report["notes"].append("reviews_with_embeddings.parquet lacks 'sku' and can't be reliably joined by 'id' → rebuild with sku.")
        except Exception as e:
            report["files"][fr] = {"exists": True, "error": repr(e), "trace": traceback.format_exc()}

    # 4) product_topic_stats (optional)
    fts = str(TOPIC_STATS)
    if not exists(TOPIC_STATS):
        report["files"][fts] = {"exists": False, "message": "Optional; used for topic boosts + labels."}
    else:
        try:
            ts = pd.read_parquet(TOPIC_STATS)
            tcols = set(ts.columns)
            has_sku = "sku" in tcols
            alias = next((c for c in ["asin","product_id","product","item_sku","product_sku"] if c in tcols), None)
            has_id  = any(c in tcols for c in ["id","review_id","reviewId"])
            has_topic_id = any(c in tcols for c in ["topic_id","topic","cluster","label_id","topicid","topicId"])
            has_weight   = any(c in tcols for c in ["count","n","freq","share","volume","weight"])
            # can we map to sku via id?
            joinable = False
            if not has_sku and alias is None and has_id and exists(REVIEWS):
                try:
                    idcol = next(c for c in ["id","review_id","reviewId"] if c in tcols)
                    left = ts[[idcol]].head(10000).rename(columns={idcol:"id"})
                    right = pd.read_parquet(REVIEWS, columns=["id","sku"])
                    j = left.merge(right, on="id", how="left")
                    joinable = j["sku"].notna().mean() > 0.9
                except Exception:
                    joinable = False
            topic_ok = has_topic_id and (has_sku or alias is not None or joinable) and has_weight
            report["files"][fts] = {
                "exists": True,
                "row_count": len(ts),
                "has_columns": {
                    "sku": has_sku,
                    "sku_alias": alias is not None,
                    "id": has_id,
                    "topic_id(any name)": has_topic_id,
                    "weight(any of count/n/freq/share/volume/weight)": has_weight
                },
                "sku_present_or_joinable": bool(has_sku or alias is not None or joinable),
                "status": passfail(topic_ok),
                "sample": df_sample_dict(ts, 5)
            }
            if not topic_ok:
                report["notes"].append("product_topic_stats.parquet missing 'sku' (or alias) and not reliably joinable by 'id' → add 'sku' or provide joinable id.")
        except Exception as e:
            report["files"][fts] = {"exists": True, "error": repr(e), "trace": traceback.format_exc()}

    # 5) topics_named (optional)
    tn = coalesce_topics_named()
    ftn = str(TOPICS_NAMED_LLM if exists(TOPICS_NAMED_LLM) else TOPICS_NAMED)
    if tn is None:
        report["files"][ftn] = {"exists": exists(TOPICS_NAMED_LLM) or exists(TOPICS_NAMED),
                                "message": "Optional; need topic_id + topic_label for nice names.",
                                "status": passfail(False)}
    else:
        ok = "topic_id" in tn.columns and "topic_label" in tn.columns
        report["files"][ftn] = {
            "exists": True,
            "has_columns": {"topic_id": "topic_id" in tn.columns, "topic_label": "topic_label" in tn.columns},
            "status": passfail(ok),
            "sample": df_sample_dict(tn, 5)
        }

    # 6) topic_cards (optional)
    fc = str(TOPIC_CARDS)
    if exists(TOPIC_CARDS):
        try:
            tc = pd.read_parquet(TOPIC_CARDS)
            ok = "topic_id" in tc.columns
            report["files"][fc] = {"exists": True, "has_topic_id": ok, "status": passfail(ok),
                                   "sample": df_sample_dict(tc, 3)}
            if not ok:
                report["notes"].append("topic_cards.parquet lacks 'topic_id' → cards won't join.")
        except Exception as e:
            report["files"][fc] = {"exists": True, "error": repr(e), "trace": traceback.format_exc()}
    else:
        report["files"][fc] = {"exists": False, "message": "Optional; pretty explanations."}

    # 7) aspects (optional)
    fa = str(ASPECTS_FILE)
    if exists(ASPECTS_FILE):
        try:
            a = pd.read_parquet(ASPECTS_FILE)
            has_id = "id" in a.columns
            has_aspect = "aspect" in a.columns
            has_sent = "sentiment" in a.columns or "polarity" in a.columns or "score" in a.columns
            ok = has_id and has_aspect
            report["files"][fa] = {
                "exists": True,
                "has_columns": {"id": has_id, "aspect": has_aspect, "sentiment_like": has_sent},
                "status": passfail(ok),
                "sample": df_sample_dict(a, 5)
            }
            if not ok:
                report["notes"].append("topics_with_aspects_llm.parquet needs at least 'id' and 'aspect'.")
        except Exception as e:
            report["files"][fa] = {"exists": True, "error": repr(e), "trace": traceback.format_exc()}
    else:
        report["files"][fa] = {"exists": False, "message": "Optional; for pros/cons & constraint matching."}

    # Final summary
    core_reviews_ok = report["files"].get(str(REVIEWS), {}).get("status") == "PASS"
    core_meta_ok = report["files"].get(str(P_META), {}).get("status") == "PASS"
    core_emb_ok = report["files"].get(str(P_EMB), {}).get("status") == "PASS"
    overall_ok = core_reviews_ok and core_meta_ok and core_emb_ok

    report["summary"] = {
        "core_ready": overall_ok,
        "core_requirements": {
            "reviews_merged.parquet": report["files"].get(str(REVIEWS), {}).get("status"),
            "product_emb_meta.parquet": report["files"].get(str(P_META), {}).get("status"),
            "product_emb.npy": report["files"].get(str(P_EMB), {}).get("status"),
        },
        "optional_signals": {
            "reviews_with_embeddings.parquet (sku present or joinable)": report["files"].get(str(REVIEWS_EMB), {}).get("status", "N/A"),
            "product_topic_stats.parquet (sku/alias or id join + topic_id + weight)": report["files"].get(str(TOPIC_STATS), {}).get("status", "N/A"),
            "topics_named(_llm).parquet (topic_id + topic_label)": report["files"].get(str(TOPICS_NAMED_LLM if exists(TOPICS_NAMED_LLM) else TOPICS_NAMED), {}).get("status", "N/A"),
            "topic_cards.parquet (topic_id)": report["files"].get(str(TOPIC_CARDS), {}).get("status", "N/A"),
            "topics_with_aspects_llm.parquet (id + aspect [ + sentiment ])": report["files"].get(str(ASPECTS_FILE), {}).get("status", "N/A"),
        },
        "action_items": report["notes"],
    }

    # Pretty print — safely
    print("\n=== Artifact Audit ===")
    for path, info in report["files"].items():
        print(f"\n- {path}")
        for k, v in info.items():
            if k in ("sample",):
                print(f"  {k}: {safe_dump(v)}")
            elif k == "trace":
                continue
            else:
                print(f"  {k}: {v}")
        if "trace" in info:
            print("  trace (last lines):")
            print("\n".join(info["trace"].splitlines()[-6:]))

    print("\n=== Summary ===")
    print(safe_dump(report["summary"], limit=2000))

    # Exit code: 0 if core is ready; 1 otherwise
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    audit()
