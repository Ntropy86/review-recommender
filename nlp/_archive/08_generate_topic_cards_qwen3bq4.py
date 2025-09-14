#!/usr/bin/env python3
"""
Fast, resume-safe topic cards with Ollama + ETA + meta cache.

Writes:
  - data/processed/topic_cards.jsonl   (one JSON per topic, append-only)
  - data/processed/topic_cards.parquet (rebuilt periodically)
  - data/processed/_topic_cards_meta.json (progress, avg sec/card, ETA)

Env knobs:
  OLLAMA_MODEL=qwen2.5:3b
  OLLAMA_HOST=http://localhost:11434
  MAX_QUOTES=3
  NUM_PREDICT=70
  NUM_CTX=768
  TEMPERATURE=0.0
  KEEP_ALIVE=5m
  MIN_REVIEWS_PER_TOPIC=5
  FLUSH_EVERY=50
  PROGRESS_EVERY=10
"""

import os, re, json, time, requests, pandas as pd
import pyarrow.parquet as pq

# --------- paths ---------
IN_REV   = os.getenv("IN_REV",  "data/processed/reviews_with_topic_labels_llm.parquet")
IN_TOP   = os.getenv("IN_TOP",  "data/processed/topics_with_aspects_llm.parquet")
OUT_JSONL= os.getenv("OUT_JSONL","data/processed/topic_cards.jsonl")
OUT_PARQ = os.getenv("OUT_PARQ", "data/processed/topic_cards.parquet")
META     = os.getenv("META",     "data/processed/_topic_cards_meta.json")

# --------- ollama ---------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL       = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
MAX_QUOTES  = int(os.getenv("MAX_QUOTES","3"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT","70"))
NUM_CTX     = int(os.getenv("NUM_CTX","768"))
TEMPERATURE = float(os.getenv("TEMPERATURE","0.0"))
KEEP_ALIVE  = os.getenv("KEEP_ALIVE","5m")
MIN_REVIEWS_PER_TOPIC = int(os.getenv("MIN_REVIEWS_PER_TOPIC","5"))
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY","50"))
PROGRESS_EVERY = int(os.getenv("PROGRESS_EVERY","10"))

# --------- utils ---------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def schema_cols(path: str):
    return set(pq.read_schema(path).names)

def norm(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+"," ", str(s)).strip()

def pick_quotes(texts, want=3, min_len=40, max_len=140):
    import re as _re
    pool = [t for t in texts if min_len <= len(str(t)) <= max_len] or [t for t in texts if str(t)]
    seen, out = set(), []
    for t in pool:
        k = _re.sub(r"[^a-z0-9]+"," ", str(t).lower())[:120]
        if k in seen: continue
        seen.add(k); out.append(norm(str(t)))
        if len(out) >= want: break
    return out

def call_ollama_json(prompt: str, retries: int = 3) -> dict:
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
    for attempt in range(retries):
        try:
            with requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=True, timeout=180) as r:
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
                import re as _re
                m = _re.search(r"\{.*\}", txt, flags=_re.S)
                return json.loads(m.group(0)) if m else {}
        except Exception as e:
            wait = 1.0 + 1.5*attempt
            print(f"[WARN] ollama call failed ({e}); retry in {wait:.1f}s")
            time.sleep(wait)
    return {"headline":"(untitled)","summary":"(no summary)"}

def build_card(topic_label: str, aspect: str, quotes: list, stars: list) -> dict:
    if stars:
        s = [float(x) for x in stars if pd.notna(x)]
        avg = round(sum(s)/max(1,len(s)), 2) if s else 0.0
    else:
        avg = 0.0
    prompt = f"""
Return ONLY a JSON object:
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
    obj = call_ollama_json(prompt)
    obj.setdefault("headline", topic_label[:80])
    obj.setdefault("summary", f"{topic_label} — {aspect}")
    return obj

def load_done_ids(jsonl_path: str):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(int(obj["topic_id"]))
                except Exception:
                    continue
    return done

def rebuild_parquet(jsonl_path: str, out_parq: str):
    rows = []
    if not os.path.exists(jsonl_path): return
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    if not rows: return
    df = pd.DataFrame(rows).sort_values("n_reviews", ascending=False)
    ensure_dir(out_parq); df.to_parquet(out_parq, index=False)

def hhmmss(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def save_meta(path: str, **kw):
    ensure_dir(path)
    with open(path, "w") as f: json.dump(kw, f)

# --------- main ---------
def main():
    # ping ollama
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise SystemExit(f"[ERROR] Ollama not reachable at {OLLAMA_HOST}. Start it with `ollama serve`. ({e})")

    # load reviews
    if not os.path.exists(IN_REV): raise SystemExit(f"[ERROR] Missing {IN_REV}")
    need = {"text","topic_id"}
    have = schema_cols(IN_REV)
    if not need.issubset(have):
        raise SystemExit(f"[ERROR] {IN_REV} missing required columns: {', '.join(sorted(need-have))}")
    cols = [c for c in ["id","text","stars","topic_id","topic_label","aspect"] if c in have]
    rev = pd.read_parquet(IN_REV, columns=cols).dropna(subset=["text"])

    # load topics (optional)
    if os.path.exists(IN_TOP):
        tcols = schema_cols(IN_TOP)
        keep = [c for c in ["topic_id","topic_label","aspect"] if c in tcols]
        top = pd.read_parquet(IN_TOP, columns=keep).drop_duplicates("topic_id")
    else:
        top = rev.groupby("topic_id").size().rename("n").reset_index()
        top["topic_label"] = top["topic_id"].apply(lambda x: f"Topic {x}")
        top["aspect"] = "Misc"

    # topic universe
    counts = rev.groupby("topic_id").size().rename("n").reset_index()
    counts = counts[(counts["topic_id"] != -1) & (counts["n"] >= MIN_REVIEWS_PER_TOPIC)]
    topic_ids = counts.sort_values("n", ascending=False)["topic_id"].astype(int).tolist()
    total = len(topic_ids)

    # resume
    ensure_dir(OUT_JSONL)
    done_ids = load_done_ids(OUT_JSONL)
    todo = [t for t in topic_ids if t not in done_ids]
    remaining = len(todo)
    start_ts = time.time()

    print(f"[INFO] model={MODEL} q={MAX_QUOTES} ctx={NUM_CTX} pred={NUM_PREDICT} keep_alive={KEEP_ALIVE}")
    print(f"[INFO] topics total={total} | already_done={len(done_ids)} | remaining={remaining} (skip <-1 and <{MIN_REVIEWS_PER_TOPIC})")

    if remaining == 0:
        rebuild_parquet(OUT_JSONL, OUT_PARQ)
        print(f"[OK] nothing to do; rebuilt {OUT_PARQ}")
        save_meta(META, total=total, done=len(done_ids), remaining=0, avg_sec_per_card=0, eta="00:00:00", updated_at=time.time())
        return

    times = []  # rolling seconds per card
    wrote = 0

    with open(OUT_JSONL, "a", encoding="utf-8") as fout:
        for idx, tid in enumerate(todo, 1):
            g = rev[rev["topic_id"]==tid]
            meta = top[top["topic_id"]==tid]
            label = (meta["topic_label"].iloc[0] if len(meta) else (g["topic_label"].iloc[0] if "topic_label" in g else f"Topic {tid}"))
            aspect = (meta["aspect"].iloc[0] if len(meta) else (g["aspect"].iloc[0] if "aspect" in g else "Misc"))
            quotes = pick_quotes(g["text"].astype(str).tolist(), want=MAX_QUOTES)
            stars  = g["stars"].astype(float).tolist() if "stars" in g else []

            t0 = time.perf_counter()
            card = build_card(label, aspect, quotes, stars)
            dt = time.perf_counter() - t0
            times.append(dt)
            if len(times) > 50: times.pop(0)  # keep last 50 for a stable average
            avg = sum(times)/len(times)

            row = {
                "topic_id": int(tid),
                "topic_label": label,
                "aspect": aspect,
                "n_reviews": int(len(g)),
                "avg_stars": float(round(g["stars"].mean(),2)) if "stars" in g else None,
                "headline": norm(card.get("headline","")),
                "summary": norm(card.get("summary","")),
                "keywords": card.get("keywords", []) if isinstance(card.get("keywords"), list) else [],
                "insight": norm(card.get("insight","")) if isinstance(card.get("insight",""), str) else "",
                "watchouts": norm(card.get("watchouts","")) if isinstance(card.get("watchouts",""), str) else "",
                "representative_quotes": quotes
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            wrote += 1

            # progress + ETA
            left = remaining - wrote
            eta_sec = avg * max(0, left)
            if (idx % PROGRESS_EVERY == 0) or (left == 0):
                print(f"[PROGRESS] {wrote}/{remaining} this_run | topic_id={tid} time={dt:.2f}s avg={avg:.2f}s ETA={hhmmss(eta_sec)}")
                save_meta(META,
                          total=total,
                          done=len(done_ids)+wrote,
                          remaining=left + (total - len(done_ids) - remaining),
                          avg_sec_per_card=avg,
                          eta=hhmmss(eta_sec),
                          updated_at=time.time())

            # periodic parquet rebuild
            if wrote % FLUSH_EVERY == 0:
                rebuild_parquet(OUT_JSONL, OUT_PARQ)
                print(f"[SYNC] parquet refreshed ({OUT_PARQ})")

    rebuild_parquet(OUT_JSONL, OUT_PARQ)
    avg = (sum(times)/len(times)) if times else 0.0
    save_meta(META,
              total=total,
              done=len(done_ids)+wrote,
              remaining=0,
              avg_sec_per_card=avg,
              eta="00:00:00",
              updated_at=time.time())
    print(f"[OK] wrote/updated {OUT_JSONL} and {OUT_PARQ} | avg={avg:.2f}s/card")
    print("[DONE]")
if __name__ == "__main__":
    main()
