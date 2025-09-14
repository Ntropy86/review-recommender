#!/usr/bin/env python3
import pandas as pd
import numpy as np

IN_PARQ  = "data/processed/reviews_with_topic_labels.parquet"
OUT_REV  = "data/processed/reviews_with_aspects.parquet"
OUT_ASP  = "data/processed/aspects_metrics.parquet"

# 1) load labeled topics
df = pd.read_parquet(IN_PARQ, columns=["id","stars","ts","source","text","topic_id","topic_label"]).dropna(subset=["topic_label"])

# 2) simple rules: topic_label → aspect (edit terms as you see patterns)
rules = {
    "Shipping": ["ship", "deliver", "arrive", "late", "courier", "tracking"],
    "Quality":  ["quality", "defect", "broke", "damage", "faulty", "durable", "scratch"],
    "Sizing":   ["size", "fit", "small", "large", "tight", "loose"],
    "Packaging":["package", "box", "seal", "packing"],
    "Service":  ["support", "return", "refund", "replace", "warranty", "customer service"],
}

def to_aspect(label: str) -> str:
    l = label.lower()
    best, hits = "Misc", 0
    for asp, kws in rules.items():
        score = sum(1 for k in kws if k in l)
        if score > hits:
            best, hits = asp, score
    return best

df["aspect"] = df["topic_label"].map(to_aspect)

# 3) metrics per aspect
g = df.groupby("aspect", as_index=False).agg(
    n_reviews=("id","count"),
    avg_stars=("stars","mean")
)
g["lost_rating"] = (5 - g["avg_stars"]) * g["n_reviews"]   # impact proxy

# (optional) quick trend: last 30 days vs prior 30 if you have dates
if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
    cutoff = df["ts"].max() - pd.Timedelta(days=30)
    recent = df[df["ts"] >= cutoff].groupby("aspect")["id"].count()
    prior  = df[df["ts"] <  cutoff].groupby("aspect")["id"].count()
    g = g.merge(recent.rename("n_30d"), on="aspect", how="left") \
         .merge(prior.rename("n_prior"), on="aspect", how="left")
    g[["n_30d","n_prior"]] = g[["n_30d","n_prior"]].fillna(0).astype(int)
    g["trend_delta"] = g["n_30d"] - (g["n_prior"] / max((df["ts"].max()-df["ts"].min()).days/30 - 1, 1))

# 4) save
df.to_parquet(OUT_REV, index=False)
g.sort_values("lost_rating", ascending=False).to_parquet(OUT_ASP, index=False)

print(f"[OK] wrote {OUT_REV} ({len(df):,} rows)")
print(f"[OK] wrote {OUT_ASP} (aspects={g.shape[0]})")
