# nlp/05a_cluster_sanity.py
#!/usr/bin/env python3
import time, numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans

IN_PARQUET  = "data/processed/reviews_with_embeddings.parquet"
OUT_PARQUET = "data/processed/reviews_with_clusters_sanity.parquet"

print("[LOAD] reading embeddings…")
df = pd.read_parquet(IN_PARQUET, columns=["id","source","stars","text","embedding"])

# (optional) start smaller to be sure it's working; bump later
N = min(50_000, len(df))
df = df.iloc[:N].copy()
X = np.vstack(df["embedding"].to_numpy()).astype("float32")

print(f"[INFO] rows={len(df):,} dims={X.shape[1]}")
print("[CLUSTER] MiniBatchKMeans starting (this prints progress)…")
t0 = time.time()

k = 60  # tweak later
mbk = MiniBatchKMeans(
    n_clusters=k,
    batch_size=2048,
    n_init="auto",
    max_iter=100,
    verbose=1,       # <-- prints per-iteration progress
    random_state=42
)
labels = mbk.fit_predict(X)

elapsed = time.time() - t0
df["topic_id"] = labels
df.to_parquet(OUT_PARQUET, index=False)

print(f"[OK] wrote {OUT_PARQUET}  rows={len(df):,}  clusters={len(set(labels))}  time={elapsed:.1f}s")
