#!/usr/bin/env python3
import os, json, pandas as pd

IN_REV  = os.getenv("IN_REV",  "data/processed/reviews_with_hdbscan.parquet")
IN_TOP  = os.getenv("IN_TOP",  "data/processed/topics_named_llm.parquet")  # optional
OUT_TOP = os.getenv("OUT_TOP", "data/processed/topics_with_aspects_llm.parquet")
OUT_REV = os.getenv("OUT_REV", "data/processed/reviews_with_topic_labels_llm.parquet")
CACHE   = os.getenv("CACHE",   "data/processed/_aspects_llm_cache.json")

assert os.path.exists(CACHE), f"Cache not found: {CACHE}"

# 1) load cache -> aspect_df
cache = json.load(open(CACHE, "r"))
rows = [{"topic_id": int(tid), "aspect": (data or {}).get("aspect","Misc")} for tid, data in cache.items()]
aspect_df = pd.DataFrame(rows)

# 2) load topics table (or synthesize minimal)
if os.path.exists(IN_TOP):
    top = pd.read_parquet(IN_TOP)
    if "topic_label" not in top.columns:
        top["topic_label"] = top["topic_id"].apply(lambda x: f"Topic {x}")
else:
    # synthesize from reviews if needed
    rev = pd.read_parquet(IN_REV, columns=["topic_id"])
    top = rev.groupby("topic_id").size().rename("n_reviews").reset_index()
    top["topic_label"] = top["topic_id"].apply(lambda x: f"Topic {x}")

# 3) merge with suffix handling
top2 = top.merge(aspect_df, on="topic_id", how="left", suffixes=("", "_new"))
# if top already had an aspect, prefer the new one
if "aspect_new" in top2.columns:
    if "aspect" in top2.columns:
        top2["aspect"] = top2["aspect_new"].fillna(top2["aspect"])
    else:
        top2 = top2.rename(columns={"aspect_new":"aspect"})
    top2 = top2.drop(columns=[c for c in ["aspect_new"] if c in top2.columns])
# fill any missing
if "aspect" not in top2.columns:
    top2["aspect"] = "Misc"
else:
    top2["aspect"] = top2["aspect"].fillna("Misc")

os.makedirs(os.path.dirname(OUT_TOP) or ".", exist_ok=True)
top2.to_parquet(OUT_TOP, index=False)

# 4) attach aspect + label back to reviews and write
rev = pd.read_parquet(IN_REV)
label_map = top2.set_index("topic_id")[["topic_label","aspect"]].to_dict(orient="index")
rev["topic_label"] = rev["topic_id"].map(lambda t: label_map.get(t,{}).get("topic_label", f"Topic {t}"))
rev["aspect"]      = rev["topic_id"].map(lambda t: label_map.get(t,{}).get("aspect","Misc"))

os.makedirs(os.path.dirname(OUT_REV) or ".", exist_ok=True)
rev.to_parquet(OUT_REV, index=False)

print(f"[OK] wrote\n  - {OUT_TOP}\n  - {OUT_REV}")
print(rev.groupby("aspect").size().sort_values(ascending=False).head(10))
