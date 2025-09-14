#!/usr/bin/env python3
import os, json, math, time, hashlib
import pandas as pd
from dotenv import load_dotenv

# --- guardrails & config ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL   = os.getenv("LLM_MODEL", "gpt-4o-mini")
DRY_RUN = os.getenv("LLM_DRY_RUN", "1") == "1"
MAX_TOPICS = int(os.getenv("LLM_MAX_TOPICS", "60"))
EVIDENCE_PER_TOPIC = int(os.getenv("LLM_EVIDENCE_PER_TOPIC", "12"))
CHAR_BUDGET = int(os.getenv("LLM_CHAR_BUDGET", "120000"))  # ~chars; ~4 chars per token

IN_PARQ  = "data/processed/reviews_with_hdbscan.parquet"
OUT_TOP  = "data/processed/topics_named_llm.parquet"
OUT_REV  = "data/processed/reviews_with_topic_labels_llm.parquet"
CACHE_F  = ".cache/topic_labels_llm.json"

# --- lightweight cache ---
if os.path.exists(CACHE_F):
    with open(CACHE_F, "r") as f:
        CACHE = json.load(f)
else:
    CACHE = {}

def cache_key(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:32]

def approx_tokens(s: str) -> int:
    # ~4 chars per token rule-of-thumb
    return max(1, math.ceil(len(s) / 4))

def call_llm(prompt: str) -> str:
    if DRY_RUN or not API_KEY:
        return "[DRY RUN] LABEL: misc; REASONS: (skipped API)"
    # lazy import to avoid dependency if not calling
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You label customer review clusters succinctly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

# --- load clustered reviews ---
cols = ["id","text","stars","source","topic_id"]
df = pd.read_parquet(IN_PARQ, columns=cols).dropna(subset=["text"])
df = df[df["topic_id"] != -1].copy()

# order topics by size (largest first) & cap
topic_sizes = df.groupby("topic_id")["id"].count().sort_values(ascending=False)
topics = topic_sizes.index.tolist()[:MAX_TOPICS]

results = []
spent_chars = 0

for t in topics:
    sub = df[df["topic_id"] == t].head(EVIDENCE_PER_TOPIC)
    # minimal cleaning & truncation to protect budget
    ev = [(" ".join(s.split()))[:300] for s in sub["text"].tolist()]
    key = cache_key(ev)
    if key in CACHE:
        label_text = CACHE[key]
    else:
        # build strict, short prompt
        prompt = (
            "You are labeling ONE review topic.\n"
            "Return ONLY this JSON with 3 keys: "
            '{"label": "...", "reasons": ["...","..."], "keywords": ["...","..."]}\n'
            "Rules: label ≤ 4 words, no brand names, no numbers unless necessary.\n"
            "Evidence (each is a customer quote):\n- " + "\n- ".join(ev)
        )
        est = approx_tokens(prompt)
        if spent_chars + len(prompt) > CHAR_BUDGET:
            label_text = '[BUDGET] {"label":"misc","reasons":["budget cap"],"keywords":["misc"]}'
        else:
            label_text = call_llm(prompt)
            spent_chars += len(prompt)
        CACHE[key] = label_text
        # be polite to rate limits even if small
        time.sleep(0.05)

    # parse very defensively
    label, reasons, keywords = "misc", [], []
    try:
        # allow raw text like: LABEL: X ... ; or JSON
        if label_text.strip().startswith("{"):
            import json as _json
            obj = _json.loads(label_text)
            label = str(obj.get("label","misc"))[:40]
            reasons = [str(r) for r in obj.get("reasons", [])][:3]
            keywords = [str(k) for k in obj.get("keywords", [])][:5]
        else:
            # fallback: take first line after 'LABEL:'
            low = label_text.lower()
            if "label" in low:
                label = label_text.split(":")[1].split("\n")[0].strip()[:40]
    except Exception:
        label = "misc"

    results.append({"topic_id": int(t),
                    "n_reviews": int(topic_sizes.loc[t]),
                    "label": label,
                    "reasons": reasons,
                    "keywords": keywords})

# persist cache
with open(CACHE_F, "w") as f:
    json.dump(CACHE, f)

# make a topics dataframe
topics_df = pd.DataFrame(results).sort_values("n_reviews", ascending=False)
topics_df.to_parquet(OUT_TOP, index=False)

# attach labels to reviews (only for the topics we processed)
label_map = dict(zip(topics_df["topic_id"], topics_df["label"]))
df["topic_label"] = df["topic_id"].map(label_map).fillna("misc")
df.to_parquet(OUT_REV, index=False)

print(f"[OK] topics labeled: {len(topics_df)}  (dry_run={DRY_RUN})")
print(f"[OUT] {OUT_TOP} and {OUT_REV}")
