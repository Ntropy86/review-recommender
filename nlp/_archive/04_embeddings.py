#!/usr/bin/env python3
"""
Reads reviews from DuckDB (reviews_raw), generates sentence embeddings in batches,
and writes a Parquet with columns: id, stars, ts, source, text, embedding(list<float>).

Model: all-MiniLM-L6-v2 (384-dim) → fast, good quality for clustering.
"""

import os, math
import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pyarrow as pa
import pyarrow.parquet as pq

DB_PATH = "data/reviews.duckdb"
OUT_PARQUET = "data/processed/reviews_with_embeddings.parquet"
BATCH_SIZE = 4096     # adjust for your RAM/GPU
MAX_ROWS   = None     # set e.g. 100_000 for quick runs

os.makedirs("data/processed", exist_ok=True)

# 1) Load ids + text (optionally cap rows for a first pass)
con = duckdb.connect(DB_PATH)
count = con.execute("SELECT COUNT(*) FROM reviews_raw").fetchone()[0]
if MAX_ROWS:
    count = min(count, MAX_ROWS)

print(f"[INFO] Reading {count:,} rows from reviews_raw ...")
df_iter = con.execute(f"""
    SELECT id, stars, ts, source, text
    FROM reviews_raw
    {f'LIMIT {MAX_ROWS}' if MAX_ROWS else ''}
""").fetch_df_chunk()  # streaming cursor

# 2) Prepare Parquet writer with schema (embedding = list<float>)
schema = pa.schema([
    pa.field("id", pa.string()),
    pa.field("stars", pa.int32()),
    pa.field("ts", pa.timestamp("us", tz=None)),   # already UTC
    pa.field("source", pa.string()),
    pa.field("text", pa.string()),
    pa.field("embedding", pa.list_(pa.float32()))
])
writer = pq.ParquetWriter(OUT_PARQUET, schema, compression="zstd")

# 3) Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Helper to write a pandas batch with embeddings list to Parquet
def write_batch(pdf: pd.DataFrame, embs: np.ndarray):
    table = pa.Table.from_pydict({
        "id": pdf["id"].astype(str).tolist(),
        "stars": pdf["stars"].astype("int32").tolist(),
        "ts": pd.to_datetime(pdf["ts"], utc=True, errors="coerce"),
        "source": pdf["source"].astype(str).tolist(),
        "text": pdf["text"].astype(str).tolist(),
        "embedding": [pa.array(e.astype(np.float32)) for e in embs]
    }, schema=schema)
    writer.write_table(table)

# 4) Stream rows from DuckDB in chunks and embed
buffer = []
total_written = 0

pbar = tqdm(total=count, desc="[EMBED]")
while True:
    chunk = con.fetch_df_chunk()
    if chunk is None or len(chunk) == 0:
        break
    # Accumulate and flush in BATCH_SIZE
    for _, row in chunk.iterrows():
        buffer.append(row)
        if len(buffer) >= BATCH_SIZE:
            pdf = pd.DataFrame(buffer)
            embs = model.encode(pdf["text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            write_batch(pdf, np.array(embs))
            total_written += len(pdf)
            pbar.update(len(pdf))
            buffer.clear()

# Flush tail
if buffer:
    pdf = pd.DataFrame(buffer)
    embs = model.encode(pdf["text"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    write_batch(pdf, np.array(embs))
    total_written += len(pdf)
    pbar.update(len(pdf))
    buffer.clear()

writer.close()
con.close()
pbar.close()
print(f"[OK] Wrote {total_written:,} rows to {OUT_PARQUET}")
