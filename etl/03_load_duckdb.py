#!/usr/bin/env python3
import os, duckdb, sys

MERGED_PARQUET = "data/processed/reviews_merged.parquet"
DB_PATH        = "data/reviews.duckdb"
TABLE_NAME     = "reviews_raw"   # bronze

if not os.path.exists(MERGED_PARQUET):
    print(f"ERR: Missing {MERGED_PARQUET}. Run 02_normalize_merge.py first.", file=sys.stderr)
    sys.exit(1)

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
con = duckdb.connect(DB_PATH)

# 1) Create or replace the bronze table from Parquet (idempotent)
con.execute(f"""
CREATE OR REPLACE TABLE {TABLE_NAME} AS
SELECT
  id::TEXT                        AS id,
  COALESCE(sku::TEXT, NULL)       AS sku,
  ts                               AS ts,      -- already UTC in Parquet
  stars::INTEGER                  AS stars,    -- 1..5
  text::TEXT                      AS text,
  source::TEXT                    AS source
FROM read_parquet('{MERGED_PARQUET}');
""")

# 2) Enforce uniqueness on id (DuckDB: UNIQUE index)
#   (Drop-and-recreate so re-runs are safe)
con.execute(f"DROP INDEX IF EXISTS idx_{TABLE_NAME}_id;")
con.execute(f"CREATE UNIQUE INDEX idx_{TABLE_NAME}_id ON {TABLE_NAME}(id);")

# 3) Quick sanity metrics you’ll paste in README
rowcount = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0]
date_min, date_max = con.execute(
    f"SELECT MIN(ts), MAX(ts) FROM {TABLE_NAME};"
).fetchone()

print(f"[OK] Loaded {rowcount:,} rows into {DB_PATH}:{TABLE_NAME}")
print(f"[Dates] min={date_min}  max={date_max}")

# 4) Helpful views for later steps (optional but nice)
con.execute(f"""
CREATE OR REPLACE VIEW v_star_dist AS
SELECT stars, COUNT(*) AS n
FROM {TABLE_NAME}
GROUP BY stars
ORDER BY stars;
""")

con.execute(f"""
CREATE OR REPLACE VIEW v_source_breakdown AS
SELECT source, COUNT(*) AS n
FROM {TABLE_NAME}
GROUP BY source
ORDER BY n DESC;
""")

con.close()
print("[OK] Created views: v_star_dist, v_source_breakdown")
