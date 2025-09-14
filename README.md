# 🛍️ AI-Powered Product Search Engine

[![GitHub](https://img.shields.io/badge/GitHub-ntropy86-181717?logo=github)](https://github.com/ntropy86)

A **hybrid search engine** that lets you query millions of product reviews in plain English.  
It combines **semantic embeddings** and **BM25 sparse retrieval** to return the most relevant products, with a professional **Streamlit app** for demo.

---

## 🚀 Features
- **Natural language search** (e.g., *"best socks that are super comfortable"*).
- **Hybrid retrieval**: dense product embeddings (`.npy`) + BM25 index (`.pkl`).
- **Review-aware ranking**: integrates product ratings & review counts.
- **Interactive UI**: modern Streamlit app with explanation tab for recruiters.

---

## 🛠️ Project Structure
```
data/processed/
  ├── product_bm25.pkl              # BM25 index
  ├── product_emb.npy                # Product embeddings
  ├── product_emb_meta.parquet       # Embedding metadata
  ├── products.parquet               # Clean product data
  ├── reviews_merged.parquet         # Reviews joined with products
  └── reviews_with_embeddings.parquet# Reviews + embeddings

nlp/
  ├── 10_product_prep.py             # Product preprocessing
  ├── 11_build_product_embeddings.py # Embedding builder
  └── 12_product_prep.py             # Review preprocessing

app/
  ├── app_product_search.py          # Streamlit app
  └── test.py                        # CLI search tester
```

---

## 📦 Installation & Usage

### 1. Clone repo
```bash
git clone https://github.com/ntropy86/product-search-engine.git
cd product-search-engine
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run preprocessing (if starting fresh)
```bash
python nlp/10_product_prep.py
python nlp/12_product_prep.py
python nlp/11_build_product_embeddings.py
```

### 4. Start the Streamlit app
```bash
streamlit run app/app_product_search.py
```

### 5. CLI testing
```bash
python app/test.py -q "best wireless headphones" -k 10
```

---

## ⚙️ How It Works

This project demonstrates how modern **Information Retrieval (IR)** systems combine **classical search** with **AI-driven embeddings** for best results:

1. **Preprocessing**
   - Raw product and review data is cleaned, normalized, and merged.
   - Each product is linked with all its associated reviews.

2. **Sparse Retrieval (BM25)**
   - A BM25 index (`product_bm25.pkl`) is built over product titles & descriptions.
   - Ensures **keyword precision** (e.g., "socks" must really appear).

3. **Dense Retrieval (Embeddings)**
   - Product text is encoded into high-dimensional vectors (`product_emb.npy`).
   - Similar queries and items are found via **cosine similarity**.

4. **Hybrid Search**
   - Results from BM25 and embeddings are fused for **breadth + precision**.
   - Example: Query = "yellow kitten socks" → BM25 ensures *socks*, embeddings connect *kitten ↔ cat motif*.

5. **Ranking Layer**
   - Candidates are reranked using:
     - Review count (trust signal)
     - Average rating (quality signal)
     - Text match boosts for attributes (color, motif, category)

6. **Streamlit UI**
   - Provides a professional, interactive demo.
   - Includes a **"How It Works" tab** explaining the pipeline clearly for recruiters.
   - Displays search results, relevance scores, and metadata.

This end-to-end pipeline mirrors **real-world e‑commerce search engines** (Amazon, Etsy) and demonstrates **practical ML + systems skills**.

---

## 🔮 Future Work
- ✅ Cross-encoder **reranker** for higher accuracy.  
- ✅ **Attribute extraction** (color, category, motif) to filter results.  
- ✅ **Synonym & typo handling** for better recall.  
- ✅ **Visual embeddings** for product images.  
- ✅ Deployment on **Hugging Face Spaces / AWS Lambda**.  
- ✅ **Evaluation metrics dashboard** (nDCG, Recall@K).

---

## 👨‍💻 Author
Built with ❤️ by [**Nitigya Kargeti**](https://github.com/ntropy86)  
Master’s in Data Science · Systems Builder · Backend & ML Engineer  

---
