#!/usr/bin/env python3
import os, json, time, random, re
import requests
import pandas as pd
from collections import Counter

# ---------------- Paths ----------------
IN_REV   = os.getenv("IN_REV",  "data/processed/reviews_with_hdbscan.parquet")
IN_TOP   = os.getenv("IN_TOP",  "data/processed/topics_named_llm.parquet")  # optional for topic_label
OUT_TOP  = os.getenv("OUT_TOP", "data/processed/topics_with_aspects_llm.parquet")
OUT_REV  = os.getenv("OUT_REV", "data/processed/reviews_with_topic_labels_llm.parquet")
CACHE    = os.getenv("CACHE",   "data/processed/_aspects_llm_cache.json")

DRY_RUN_TOP_N = int(os.getenv("DRY_RUN_TOP_N", "0"))

# ---------------- Ollama ----------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
LLM_VOTES   = int(os.getenv("LLM_VOTES", "3"))
LLM_TEMP    = float(os.getenv("LLM_TEMP", "0.2"))

ASPECTS = ["Shipping","Quality","Sizing","Packaging","Service","Pricing","Usability","Misc"]
ASPECT_DEFS = {
    "Shipping":  "delivery time, tracking, couriers, delays, arrival condition from transit",
    "Quality":   "build/material durability, defects, workmanship, finish, reliability",
    "Sizing":    "fit, measurements, runs small/large, true-to-size issues",
    "Packaging": "box/packaging condition, seals, protective wrap, damaged on arrival (from packaging)",
    "Service":   "support interactions: returns, refunds, replacements, agent response",
    "Pricing":   "cost, value for money, discounts, overpriced/cheap",
    "Usability": "setup, instructions, software/app, pairing, battery, updates, UI/UX, bugs",
    "Misc":      "does not clearly belong to any other aspect"
}

# ------------- Helpers -------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def ping_ollama() -> None:
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"[ERROR] Ollama not reachable at {OLLAMA_HOST}. Start it with `ollama serve`. ({e})")

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s

def pick_quotes(texts, want=8, min_len=40, max_len=200):
    pool = [t for t in texts if min_len <= len(t) <= max_len]
    if len(pool) < want:
        pool = [t for t in texts if len(t) > 0]
    # light dedup by normalized prefix
    seen, reps = set(), []
    for t in pool:
        k = re.sub(r"[^a-z0-9]+"," ", t.lower())[:120]
        if k in seen: continue
        seen.add(k)
        reps.append(t.strip())
        if len(reps) >= want: break
    return reps

def ask_once(quotes):
    """
    One LLM vote. Returns dict: {aspect, evidence_phrases, rationale}
    """
    system = (
        "You are an annotation assistant. You must choose exactly ONE aspect for the topic "
        "from a fixed taxonomy. Use the definitions to disambiguate. "
        "Return ONLY a single JSON object, no extra text."
    )
    taxonomy = "\n".join([f"- {k}: {v}" for k,v in ASPECT_DEFS.items()])
    prompt = f"""
Choose EXACTLY ONE aspect from this taxonomy and return strict JSON:
{{
  "aspect": "<one of {ASPECTS}>",
  "evidence_phrases": ["<short phrase 1>", "<short phrase 2>"],
  "rationale": "<<=15 words>"
}}

Taxonomy:
{taxonomy}

Quotes (noisy, may mix sentiments):
{json.dumps(quotes, ensure_ascii=False, indent=2)}

Rules:
- Aspect MUST be one of {ASPECTS}.
- Prefer specific aspects over Misc.
- If quotes are about delivery time, tracking, couriers, or delay → Shipping.
- If about product defects/material/durability → Quality.
- If about fit/size → Sizing.
- If about packaging/box/seal → Packaging.
- If about customer support/returns/refunds → Service.
- If about cost/value/discounts → Pricing.
- If about setup/app/UX/battery/bugs → Usability.
- Otherwise → Misc.
Only return the JSON object.
""".strip()

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "format": "json",
        "system": system,
        "stream": True,
        "options": {
            "temperature": LLM_TEMP,
            # vary seed for each vote to get diversity even at low temperature
            "seed": random.randint(1, 2_000_000_000)
        },
    }

    with requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        chunks = []
        for line in r.iter_lines():
            if not line: continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                chunks.append(data["response"])
            if data.get("done"): break
    txt = "".join(chunks).strip()

    # robust JSON parse
    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        obj = json.loads(m.group(0)) if m else {}
    aspect = str(obj.get("aspect","Misc")).strip()
    if aspect not in ASPECTS:
        aspect = "Misc"
    ev = obj.get("evidence_phrases", [])
    if not isinstance(ev, list):
        ev = []
    ev = [normalize_text(x)[:60] for x in ev][:3]
    rat = normalize_text(obj.get("rationale",""))[:80]
    return {"aspect": aspect, "evidence_phrases": ev, "rationale": rat}

def vote_aspect(quotes, votes=3):
    ballots = [ask_once(quotes) for _ in range(max(1, votes))]
    tally = Counter([b["aspect"] for b in ballots])
    # prefer a non-Misc tie if exists
    most_common = tally.most_common()
    chosen = most_common[0][0]
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        # tie: pick non-Misc if possible
        for asp, _ in most_common:
            if asp != "Misc":
                chosen = asp
                break
    return chosen, ballots

# ------------- Load data -------------
ping_ollama()

rev_cols = ["id","text","topic_id","stars","source"]
rev = pd.read_parquet(IN_REV, columns=rev_cols).dropna(subset=["text"])
rev = rev[rev["topic_id"] != -1].copy()
rev["text"] = rev["text"].astype(str).map(normalize_text)

# Optional topics table (for existing labels)
if os.path.exists(IN_TOP):
    top = pd.read_parquet(IN_TOP)
    if "topic_label" not in top.columns:
        top["topic_label"] = top["topic_id"].apply(lambda x: f"Topic {x}")
else:
    top = (rev.groupby("topic_id").size().rename("n_reviews").reset_index())
    top["topic_label"] = top["topic_id"].apply(lambda x: f"Topic {x}")

# Cache
cache = {}
if os.path.exists(CACHE):
    try:
        cache = json.load(open(CACHE, "r"))
    except Exception:
        cache = {}

# ------------- Main loop -------------
rows = []
topics = sorted(rev["topic_id"].unique().tolist())
if DRY_RUN_TOP_N > 0:
    topics = topics[:DRY_RUN_TOP_N]

print(f"[INFO] topics to classify (LLM-only): {len(topics)} using {MODEL}, votes={LLM_VOTES}, temp={LLM_TEMP}")

for i, tid in enumerate(topics, 1):
    key = str(tid)
    g = rev[rev["topic_id"] == tid]
    texts = g["text"].tolist()
    quotes = pick_quotes(texts, want=8)

    if key in cache:
        out = cache[key]
        aspect = out["aspect"]
        ballots = out.get("ballots", [])
    else:
        aspect, ballots = vote_aspect(quotes, votes=LLM_VOTES)
        out = {"aspect": aspect, "ballots": ballots}
        cache[key] = out
        # periodic cache flush
        if i % 10 == 0:
            ensure_dir(CACHE)
            with open(CACHE, "w") as f:
                json.dump(cache, f, ensure_ascii=False)

    rows.append({"topic_id": int(tid),
                 "aspect": aspect,
                 "evidence_sample": quotes[:3],
                 "ballots": ballots})

    if i % 10 == 0 or i == len(topics):
        print(f"[PROGRESS] {i}/{len(topics)} topics; last -> aspect={aspect}")

# ------------- Write outputs -------------
aspect_df = pd.DataFrame(rows)

# Merge with topic labels (if any)
top2 = top.merge(aspect_df[["topic_id","aspect"]], on="topic_id", how="left")
top2["aspect"] = top2["aspect"].fillna("Misc")

ensure_dir(OUT_TOP)
top2.to_parquet(OUT_TOP, index=False)

# Attach aspect to reviews; keep any prior topic_label if present
label_map = top2.set_index("topic_id")[["topic_label","aspect"]].to_dict(orient="index")
rev["topic_label"] = rev["topic_id"].map(lambda t: label_map.get(t,{}).get("topic_label", f"Topic {t}"))
rev["aspect"]      = rev["topic_id"].map(lambda t: label_map.get(t,{}).get("aspect","Misc"))

ensure_dir(OUT_REV)
rev.to_parquet(OUT_REV, index=False)

# Final cache flush
ensure_dir(CACHE)
with open(CACHE, "w") as f:
    json.dump(cache, f, ensure_ascii=False)

print(f"[OK] wrote {OUT_TOP} and {OUT_REV}")
print("\n[ASPECT COVERAGE]")
print(rev.groupby("aspect").size().sort_values(ascending=False))
