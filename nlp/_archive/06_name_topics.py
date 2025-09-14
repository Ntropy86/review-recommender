#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

IN_PARQUET  = "data/processed/reviews_with_hdbscan.parquet"
OUT_TOPICS  = "data/processed/topics_named.parquet"
OUT_REVIEWS = "data/processed/reviews_with_topic_labels.parquet"

# 1) load clustered reviews
cols = ["id","text","stars","source","topic_id"]
df = pd.read_parquet(IN_PARQUET, columns=cols).dropna(subset=["text"])

# ignore noise (HDBSCAN gives -1 for unclustered points)
df = df[df["topic_id"] != -1].copy()

# 2) fit TF-IDF on all texts (unigrams + bigrams)
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1,2),
    stop_words="english",
    min_df=3
)
X = vectorizer.fit_transform(df["text"])
vocab = np.array(vectorizer.get_feature_names_out())

# 3) build per-cluster keywords
topics = []
for t in sorted(df["topic_id"].unique()):
    mask = (df["topic_id"] == t).to_numpy()
    if mask.sum() == 0: 
        continue

    x_mean = X[mask].mean(axis=0)      # average TF-IDF per term
    x_mean = np.asarray(x_mean).ravel()
    top_idx = x_mean.argsort()[::-1][:8]
    top_terms = vocab[top_idx].tolist()

    examples = (
        df.loc[mask,"text"]
          .head(3)
          .str.replace(r"\s+"," ",regex=True)
          .str.slice(0,150)
          .tolist()
    )

    topics.append({
        "topic_id": int(t),
        "n_reviews": int(mask.sum()),
        "top_terms": top_terms,
        "examples": examples
    })

topics_df = pd.DataFrame(topics).sort_values("n_reviews", ascending=False)
topics_df.to_parquet(OUT_TOPICS, index=False)

# 4) create quick human-readable label (top 3 terms joined)
label_map = {row.topic_id: " / ".join(row.top_terms[:3]) 
             for _,row in topics_df.iterrows()}
df["topic_label"] = df["topic_id"].map(label_map)

df.to_parquet(OUT_REVIEWS, index=False)
print(f"[OK] wrote {OUT_TOPICS} and {OUT_REVIEWS}")
