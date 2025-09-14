#!/usr/bin/env python3
# Usage:
#   export OLLAMA_MODEL=qwen2.5:3b            # or your Q4-tagged variant
#   export MAX_QUOTES=3 NUM_PREDICT=70 NUM_CTX=768
#   python3 nlp/08b_benchmark_cards_fast.py
#
# Notes:
# - This only benchmarks (does NOT write cards).
# - Excludes topic_id == -1 and tiny topics by default.

import os, time, json, re, random, requests, pandas as pd
import pyarrow.parquet as pq
random.seed(42)

# -------- Inputs / knobs --------
IN_REV   = os.getenv("IN_REV",  "data/processed/reviews_with_topic_labels_llm.parquet")
IN_TOP   = os.getenv("IN_TOP",  "data/processed/topics_with_aspects_llm.parquet")
EXISTING = os.getenv("CARDS_PQ","data/processed/topic_cards.parquet")  # optional: counts already-done
MODEL    = os.getenv("OLLAMA_MODEL","qwen2.5:3b")
OLLAMA   = os.getenv("OLLAMA_HOST","http://localhost:11434")

# Speed knobs (defaults chosen to target <30min on Apple Silicon with 3B Q4)
MAX_QUOTES  = int(os.getenv("MAX_QUOTES","3"))     # fewer quotes == faster
NUM_PREDICT = int(os.getenv("NUM_PREDICT","70"))   # limit output tokens
NUM_CTX     = int(os.getenv("NUM_CTX","768"))      # small context fits prompt
TEMPERATURE = float(os.getenv("TEMPERATURE","0.0"))
KEEP_ALIVE  = os.getenv("KEEP_ALIVE","5m")         # keep model loaded
SAMPLE_TOPICS = int(os.getenv("SAMPLE_TOPICS","24"))  # how many topics to time
MIN_REVIEWS_PER_TOPIC = int(os.getenv("MIN_REVIEWS_PER_TOPIC","5"))

# -------- helpers --------
def schema_cols(path: str):
    if not os.path.exists(path): raise FileNotFoundError(path)
    return set(pq.read_schema(path).names)

def norm(s: str) -> str:
    return re.sub(r"\s+"," ", str(s)).strip()

def pick_quotes(texts, want=3, min_len=40, max_len=140):
    pool = [t for t in texts if min_len <= len(str(t)) <= max_len] or [t for t in texts if str(t)]
    seen, out = set(), []
    for t in pool:
        k = re.sub(r"[^a-z0-9]+"," ", str(t).lower())[:120]
        if k in seen: continue
        seen.add(k); out.append(norm(str(t)))
        if len(out) >= want: break
    return out

def ollama_json(prompt: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX
        }
    }
    with requests.post(f"{OLLAMA}/api/generate", json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        chunks=[]
        for line in r.iter_lines():
            if not line: continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data: chunks.append(data["response"])
            if data.get("done"): break
    txt = "".join(chunks).strip()
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        return json.loads(m.group(0)) if m else {}

def build_card(topic_label: str, aspect: str, quotes: list, stars: list) -> dict:
    # very small schema for speed
    if stars:
        s = [float(x) for x in stars if pd.notna(x)]
        avg = round(sum(s)/max(1,len(s)), 2) if s else 0.0
    else:
        avg = 0.0
    prompt = f"""
Return ONLY a JSON object with keys:
{{
  "headline": "6–9 words",
  "summary": "one sentence, <=20 words"
}}
Context:
- Topic label: {topic_label}
- Aspect: {aspect}
- Avg stars: {avg}
- Quotes:
{json.dumps(quotes, ensure_ascii=False, indent=2)}
""".strip()
    obj = ollama_json(prompt)
    obj.setdefault("headline", topic_label[:80])
    obj.setdefault("summary", f"{topic_label} — {aspect}")
    return obj

def hhmmss(seconds: float) -> str:
    s = int(round(seconds)); h=s//3600; m=(s%3600)//60; return f"{h:02d}:{m:02d}:{s%60:02d}"

# -------- main --------
def main():
    # ping
    try:
        requests.get(f"{OLLAMA}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"[ERROR] Ollama not reachable at {OLLAMA}. Start it with `ollama serve`. ({e})")

    # load reviews (only needed cols)
    rev_avail = schema_cols(IN_REV)
    need = {"text","topic_id"}
    if not need.issubset(rev_avail):
        missing = ", ".join(sorted(need - rev_avail))
        raise SystemExit(f"[ERROR] {IN_REV} missing required columns: {missing}")
    cols = [c for c in ["id","text","stars","topic_id","topic_label","aspect","source"] if c in rev_avail]
    rev = pd.read_parquet(IN_REV, columns=cols).dropna(subset=["text"])

    # load topics (optional)
    if os.path.exists(IN_TOP):
        tcols_avail = schema_cols(IN_TOP)
        tcols = [c for c in ["topic_id","topic_label","aspect"] if c in tcols_avail]
        top = pd.read_parquet(IN_TOP, columns=tcols).drop_duplicates("topic_id")
    else:
        top = rev.groupby("topic_id").size().rename("n").reset_index()
        top["topic_label"] = top["topic_id"].apply(lambda x: f"Topic {x}")
        top["aspect"] = "Misc"

    # filter topics: exclude noise (-1) and tiny clusters
    counts = rev.groupby("topic_id").size().rename("n").reset_index()
    counts = counts[(counts["topic_id"] != -1) & (counts["n"] >= MIN_REVIEWS_PER_TOPIC)]
    topic_ids = counts["topic_id"].tolist()
    total_topics = len(topic_ids)

    # existing cards (optional)
    done = 0
    if os.path.exists(EXISTING):
        try:
            done = int(pd.read_parquet(EXISTING, columns=["topic_id"])["topic_id"].nunique())
        except Exception:
            done = 0
    remaining = max(0, total_topics - done)

    if remaining == 0:
        print("[INFO] All topics already have cards (or nothing to do).")
        return

    # sample topics to time (largest first helps worst-case)
    counts = counts.sort_values("n", ascending=False)
    sample = counts["topic_id"].head(SAMPLE_TOPICS).tolist()

    # warm-up (loads model)
    _tid = sample[0]
    g = rev[rev["topic_id"]==_tid]
    meta = top[top["topic_id"]==_tid]
    label = (meta["topic_label"].iloc[0] if len(meta) else (g["topic_label"].iloc[0] if "topic_label" in g else f"Topic {_tid}"))
    aspect = (meta["aspect"].iloc[0] if len(meta) else (g["aspect"].iloc[0] if "aspect" in g else "Misc"))
    _ = build_card(label, aspect, pick_quotes(g["text"].astype(str).tolist(), want=MAX_QUOTES), g["stars"].tolist() if "stars" in g else [])

    # time the sample
    times = []
    for tid in sample:
        g = rev[rev["topic_id"]==tid]
        meta = top[top["topic_id"]==tid]
        label = (meta["topic_label"].iloc[0] if len(meta) else (g["topic_label"].iloc[0] if "topic_label" in g else f"Topic {tid}"))
        aspect = (meta["aspect"].iloc[0] if len(meta) else (g["aspect"].iloc[0] if "aspect" in g else "Misc"))
        quotes = pick_quotes(g["text"].astype(str).tolist(), want=MAX_QUOTES)
        stars  = g["stars"].tolist() if "stars" in g else []
        t0 = time.time()
        _ = build_card(label, aspect, quotes, stars)
        times.append(time.time() - t0)

    per_card = sum(times) / max(1, len(times))
    eta_sec = per_card * remaining

    print("\n=== Fast benchmark (3B, Q4-friendly settings) ===")
    print(f"Model: {MODEL} | Quotes: {MAX_QUOTES} | num_predict: {NUM_PREDICT} | num_ctx: {NUM_CTX} | keep_alive: {KEEP_ALIVE}")
    print(f"Topics total (filtered): {total_topics} | Done: {done} | Remaining: {remaining}")
    print(f"Sampled {len(sample)} topics -> avg per-card: {per_card:.2f} sec")
    print(f"Estimated wall time for remaining: ~{hhmmss(eta_sec)}")
    if eta_sec <= 30*60:
        print("✅ Likely under 30 minutes.")
    else:
        print("⚠️ Over 30 minutes. Reduce MAX_QUOTES / NUM_PREDICT, or batch multiple cards per call.")

if __name__ == "__main__":
    main()
