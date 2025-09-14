#!/usr/bin/env python3
import time, numpy as np, pandas as pd
import pyarrow.parquet as pq
from umap import UMAP
import hdbscan

IN_PARQ  = "data/processed/reviews_with_embeddings.parquet"
OUT_PARQ = "data/processed/reviews_with_hdbscan.parquet"

# ---- tunables (start conservative; adjust after first run) ----
MAX_ROWS        = 300000        # e.g. 120_000 for a quicker first pass
UMAP_N_COMPONENTS = 50        # 50D usually good for HDBSCAN
UMAP_N_NEIGHBORS  = 15
MIN_CLUSTER_SIZE  = 40        # raise for fewer/larger clusters
MIN_SAMPLES       = 10

print("[LOAD] embeddings…")
df = pd.read_parquet(IN_PARQ, columns=["id","text","stars","source","embedding"])
if MAX_ROWS:
    df = df.iloc[:MAX_ROWS].copy()
X = np.vstack(df["embedding"].to_numpy()).astype("float32")
print(f"[INFO] rows={len(df):,} dims={X.shape[1]}")

# ---- speedup 1: reduce dims with UMAP (uses PyNNDescent under the hood) ----
t0 = time.time()
print("[UMAP] reducing to", UMAP_N_COMPONENTS, "dimensions…")
umap = UMAP(
    n_components=UMAP_N_COMPONENTS,
    n_neighbors=UMAP_N_NEIGHBORS,
    metric="euclidean",
    min_dist=0.0,
    random_state=42,
    verbose=True,
)
Xr = umap.fit_transform(X)
print(f"[UMAP] done in {time.time()-t0:.1f}s  ->  shape={Xr.shape}")

# ---- speedup 2: HDBSCAN on reduced space, with approximate MST ----
t1 = time.time()
print("[HDBSCAN] clustering…")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    core_dist_n_jobs=0,          # use all cores
    approx_min_span_tree=True,   # faster
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(Xr)
elapsed = time.time() - t1
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = int((labels == -1).sum())
print(f"[HDBSCAN] clusters={n_clusters}  noise={noise}  time={elapsed:.1f}s")

df["topic_id"] = labels
df.to_parquet(OUT_PARQ, index=False)
print(f"[OK] wrote {OUT_PARQ}  rows={len(df):,}")
