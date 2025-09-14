#!/usr/bin/env python3
import os, json, time, random, pandas as pd
from dotenv import load_dotenv
from groq import Groq
from httpx import HTTPStatusError

load_dotenv()

IN_PARQ  = "data/processed/reviews_with_hdbscan.parquet"
OUT_TOP  = "data/processed/topics_named_llm.parquet"
OUT_REV  = "data/processed/reviews_with_topic_labels_llm.parquet"
CACHE    = "data/processed/_llm_topic_cache.json"

# IMPORTANT: use a valid model name for your account
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_EXAMPLES_PER_TOPIC = int(os.getenv("MAX_EXAMPLES_PER_TOPIC", "12"))  # smaller = cheaper & faster
TEMPERATURE = 0.0

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
assert client.api_key, "Missing GROQ_API_KEY in .env"

# Load clustered reviews
cols = ["id","text","topic_id","stars","source"]
df = pd.read_parquet(IN_PARQ, columns=cols).dropna(subset=["text"])
df = df[df["topic_id"] != -1].copy()

# Cache (resume-safe)
cache = {}
if os.path.exists(CACHE):
    try:
        cache = json.load(open(CACHE))
    except Exception:
        cache = {}

def ask_llm_with_retry(examples_text: str, max_retries: int = 8):
    """
    Robust call with exponential backoff for 429 & transient errors.
    """
    base = 1.5  # seconds
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role":"user",
                    "content": f"""
Return STRICT JSON with keys EXACTLY: "topic_label", "aspect", "rationale".
ASPECT must be one of: ["Shipping","Quality","Sizing","Packaging","Service","Pricing","Usability","Misc"].
Keep rationale <= 30 words. Use only the quotes below.

EXAMPLES:
{examples_text}
""".strip()
                }],
                temperature=TEMPERATURE,
                response_format={"type":"json_object"},
            )
            return resp.choices[0].message.content
        except HTTPStatusError as e:
            # 429 or similar transient -> backoff
            status = e.response.status_code if e.response else None
            if status == 429 or 500 <= status < 600:
                wait = base * (2 ** attempt) + random.uniform(0, 0.7)
                # Respect Retry-After if provided
                try:
                    ra = e.response.headers.get("Retry-After")
                    if ra:
                        wait = max(wait, float(ra))
                except Exception:
                    pass
                print(f"[LLM] {status} backoff {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            # network hiccup, parse error before return, etc. -> small backoff
            wait = 1.0 + random.uniform(0, 0.5)
            print(f"[LLM] transient error: {e} -> sleep {wait:.1f}s")
            time.sleep(wait)
    # if we exhausted retries, return a fallback
    return json.dumps({"topic_label":"Misc","aspect":"Misc","rationale":"retry_exhausted"})

rows = []
processed = 0
total_topics = df["topic_id"].nunique()
print(f"[INFO] topics to label: {total_topics}")

for tid, g in df.groupby("topic_id", sort=True):
    key = str(tid)
    if key in cache:
        meta = cache[key]
    else:
        # Collect up to N concise quotes
        ex = (
            g["text"].head(MAX_EXAMPLES_PER_TOPIC)
             .str.replace(r"\s+"," ", regex=True)
             .str.slice(0, 160)
             .tolist()
        )
        examples_text = "\n".join(f'- "{t}"' for t in ex)
        raw = ask_llm_with_retry(examples_text)
        try:
            meta = json.loads(raw)
            # Defensive normalization
            tl = str(meta.get("topic_label","Misc")).strip() or "Misc"
            asp = str(meta.get("aspect","Misc")).strip()
            if asp not in {"Shipping","Quality","Sizing","Packaging","Service","Pricing","Usability","Misc"}:
                asp = "Misc"
            rat = str(meta.get("rationale","")).strip()[:200]
            meta = {"topic_label": tl, "aspect": asp, "rationale": rat}
        except Exception:
            meta = {"topic_label":"Misc","aspect":"Misc","rationale":"parse_error"}
        cache[key] = meta

        # Light pacing to avoid bursts (tune if you still hit 429)
        time.sleep(0.3)

        # Flush cache periodically (resume-safe)
        if processed % 10 == 0:
            with open(CACHE, "w") as f:
                json.dump(cache, f)

    rows.append({
        "topic_id": int(tid),
        "n_reviews": int(len(g)),
        "topic_label": meta["topic_label"],
        "aspect": meta["aspect"],
        "rationale": meta.get("rationale",""),
    })
    processed += 1
    if processed % 10 == 0:
        print(f"[PROGRESS] labeled {processed}/{total_topics} topics")

# Final writes
topics_df = pd.DataFrame(rows).sort_values("n_reviews", ascending=False)
os.makedirs(os.path.dirname(OUT_TOP), exist_ok=True)
topics_df.to_parquet(OUT_TOP, index=False)

with open(CACHE, "w") as f:
    json.dump(cache, f)

# Attach back to reviews
label_map = topics_df.set_index("topic_id")[["topic_label","aspect"]].to_dict(orient="index")
df["topic_label"] = df["topic_id"].map(lambda t: label_map.get(t,{}).get("topic_label","Misc"))
df["aspect"]      = df["topic_id"].map(lambda t: label_map.get(t,{}).get("aspect","Misc"))
df.to_parquet(OUT_REV, index=False)

print(f"[OK] {OUT_TOP} and {OUT_REV}")
