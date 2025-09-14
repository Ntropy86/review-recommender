#!/usr/bin/env python3
import os, json, time, random
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --------- Paths (same as your script) ----------
IN_PARQ  = os.getenv("IN_PARQ",  "data/processed/reviews_with_hdbscan.parquet")
OUT_TOP  = os.getenv("OUT_TOP",  "data/processed/topics_named_llm.parquet")
OUT_REV  = os.getenv("OUT_REV",  "data/processed/reviews_with_topic_labels_llm.parquet")
CACHE    = os.getenv("CACHE",    "data/processed/_llm_topic_cache.json")

# --------- Model / runtime ----------------------
MODEL         = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
MAX_EXAMPLES_PER_TOPIC = int(os.getenv("MAX_EXAMPLES_PER_TOPIC", "12"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.0"))

ASPECTS = ["Shipping","Quality","Sizing","Packaging","Service","Pricing","Usability","Misc"]

# --------- Ensure Ollama is reachable -----------
def ping_ollama():
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False

def ensure_model(model: str):
    """
    Pull the model if it isn't present. Safe to call even if already pulled.
    Uses Ollama /api/pull streaming endpoint.
    """
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model}, stream=True, timeout=300)
        for line in r.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
                # Print minimal progress feedback (optional)
                if "status" in msg:
                    print(f"[MODEL] {msg['status']}", end="\r")
            except Exception:
                pass
    except requests.RequestException as e:
        print(f"[WARN] Could not auto-pull model '{model}': {e}")

# --------- LLM call via /api/generate -----------
def ask_llm_with_retry(examples_text: str, max_retries: int = 8) -> str:
    """
    Calls Ollama locally with streaming; aggressively asks for strict JSON.
    Retries on transient errors. Returns the raw text from the model.
    """
    system_msg = (
        "You are a precise classification assistant. "
        "You MUST reply with a single valid JSON object and nothing else. "
        'Keys: "topic_label", "aspect", "rationale". '
        f'Aspect must be one of: {ASPECTS}. '
        "Rationale <= 30 words."
    )
    user_prompt = f"""
Return STRICT JSON with keys EXACTLY: "topic_label", "aspect", "rationale".
ASPECT must be one of: {ASPECTS}.
Keep rationale <= 30 words. Use only the quotes below.

EXAMPLES:
{examples_text}

Output only the JSON object. No backticks, no extra text.
""".strip()

    payload = {
        "model": MODEL,
        "prompt": user_prompt,
        # Helps many models stick to valid JSON
        "format": "json",
        "system": system_msg,
        "stream": True,
        "options": {"temperature": TEMPERATURE},
    }

    base = 1.5
    for attempt in range(max_retries):
        try:
            with requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                chunks = []
                for line in r.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        chunks.append(data["response"])
                    if data.get("done"):
                        break
                return "".join(chunks)
        except requests.HTTPError as e:
            # Local Ollama rarely returns 429; still backoff on 5xx.
            status = e.response.status_code if e.response is not None else None
            if status and (status == 429 or 500 <= status < 600):
                wait = base * (2 ** attempt) + random.uniform(0, 0.7)
                print(f"[LLM] HTTP {status}, backoff {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            wait = 1.0 + random.uniform(0, 0.5)
            print(f"[LLM] transient error: {e} -> sleep {wait:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)

    # Fallback if all retries exhausted
    return json.dumps({"topic_label":"Misc","aspect":"Misc","rationale":"retry_exhausted"})

# --------- Load data ---------------------------
cols = ["id","text","topic_id","stars","source"]
df = pd.read_parquet(IN_PARQ, columns=cols).dropna(subset=["text"])
df = df[df["topic_id"] != -1].copy()

# --------- Cache (resume-safe) ----------------
cache = {}
if os.path.exists(CACHE):
    try:
        cache = json.load(open(CACHE))
    except Exception:
        cache = {}

# --------- Ensure Ollama ready & model present -
if not ping_ollama():
    raise SystemExit(
        f"[ERROR] Can't reach Ollama at {OLLAMA_HOST}. "
        "Start it with 'ollama serve' or 'brew services start ollama'."
    )
ensure_model(MODEL)

# --------- Main loop --------------------------
rows = []
processed = 0
total_topics = df["topic_id"].nunique()
print(f"[INFO] topics to label: {total_topics}")

for tid, g in df.groupby("topic_id", sort=True):
    key = str(tid)
    if key in cache:
        meta = cache[key]
    else:
        ex = (
            g["text"].head(MAX_EXAMPLES_PER_TOPIC)
             .str.replace(r"\s+"," ", regex=True)
             .str.slice(0, 160)  # keep prompts short & cheap
             .tolist()
        )
        examples_text = "\n".join(f'- "{t}"' for t in ex)

        raw = ask_llm_with_retry(examples_text)
        try:
            meta = json.loads(raw)
            tl  = str(meta.get("topic_label","Misc")).strip() or "Misc"
            asp = str(meta.get("aspect","Misc")).strip()
            if asp not in ASPECTS:
                asp = "Misc"
            rat = str(meta.get("rationale","")).strip()[:200]
            meta = {"topic_label": tl, "aspect": asp, "rationale": rat}
        except Exception:
            meta = {"topic_label":"Misc","aspect":"Misc","rationale":"parse_error"}

        cache[key] = meta

        # Light pacing (tune as needed)
        time.sleep(0.15)

        # Periodic cache flush
        if processed % 10 == 0:
            os.makedirs(os.path.dirname(CACHE) or ".", exist_ok=True)
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

# --------- Final writes -----------------------
topics_df = pd.DataFrame(rows).sort_values("n_reviews", ascending=False)
os.makedirs(os.path.dirname(OUT_TOP) or ".", exist_ok=True)
topics_df.to_parquet(OUT_TOP, index=False)

with open(CACHE, "w") as f:
    json.dump(cache, f)

label_map = topics_df.set_index("topic_id")[["topic_label","aspect"]].to_dict(orient="index")
df["topic_label"] = df["topic_id"].map(lambda t: label_map.get(t,{}).get("topic_label","Misc"))
df["aspect"]      = df["topic_id"].map(lambda t: label_map.get(t,{}).get("aspect","Misc"))
os.makedirs(os.path.dirname(OUT_REV) or ".", exist_ok=True)
df.to_parquet(OUT_REV, index=False)

print(f"[OK] wrote {OUT_TOP} and {OUT_REV}")
