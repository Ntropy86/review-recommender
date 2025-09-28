# 🚀 Deployment Guide

This guide covers different deployment options for the Review Search Copilot application.

## 📋 Prerequisites

1. **Data Files**: Ensure you have processed data files:
   - `data/processed/product_emb.npy`
   - `data/processed/product_emb_meta.parquet`
   - `data/processed/product_bm25.pkl` (optional)
   - `data/processed/reviews_with_embeddings.parquet` (optional)

2. **Dependencies**: Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🐳 Docker Deployment (Recommended)

### Quick Start

1. **Build the image**:
   ```bash
   docker build -t review-search-app .
   ```

2. **Run with data mounted**:
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/data:/app/data:ro \
     -e ENVIRONMENT=production \
     review-search-app
   ```

### Using Docker Compose

1. **Development setup**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   docker-compose up
   ```

2. **Production setup**:
   ```bash
   cp .env.example .env
   # Set ENVIRONMENT=production in .env
   docker-compose --profile production up -d
   ```

## 🖥️ Local Development

1. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env for development settings
   ```

2. **Run tests**:
   ```bash
   python run_tests.py
   ```

3. **Start application**:
   ```bash
   ./start.sh
   # Or directly:
   streamlit run app/app_product_search.py
   ```

## ☁️ Cloud Deployment

### Hugging Face Spaces

1. **Create a new Space** on Hugging Face
2. **Upload files**:
   - All Python files
   - `requirements.txt`
   - `Dockerfile`
   - Data files (if not too large)
3. **Set environment variables** in Space settings
4. **Deploy** - HF will automatically build and deploy

### AWS/GCP/Azure

1. **Container-based deployment**:
   ```bash
   # Build and tag for your registry
   docker build -t your-registry/review-search-app .
   docker push your-registry/review-search-app
   ```

2. **Use orchestration service**:
   - AWS: ECS Fargate, EKS
   - GCP: Cloud Run, GKE
   - Azure: Container Instances, AKS

### Railway/Render/Fly.io

1. **Connect your repository**
2. **Set environment variables**
3. **Deploy** - platform will auto-detect Dockerfile

## 🔧 Configuration

### Environment Variables

Key environment variables to set:

```bash
# Required
ENVIRONMENT=production
DATA_DIR=data/processed

# Optional
EMB_MODEL=BAAI/bge-small-en-v1.5
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LOG_LEVEL=INFO
MAX_REVIEWS_SCAN=300000

# Feature flags
ENABLE_BM25=true
ENABLE_RERANKING=true
ENABLE_SNIPPETS=true
```

### Production Settings

For production deployment:

1. **Set secure configuration**:
   ```bash
   ENVIRONMENT=production
   LOG_LEVEL=INFO
   HTTPS_ONLY=true
   ```

2. **Resource limits** (in docker-compose or k8s):
   - Memory: 4-8GB (for embedding models)
   - CPU: 2-4 cores
   - Storage: 2-10GB (for data files)

3. **Health checks**:
   ```bash
   python health_check.py
   ```

## 🔍 Monitoring & Debugging

### Health Checks

- **Application**: `GET /` should return 200
- **Data**: `python health_check.py`
- **Dependencies**: `python run_tests.py`

### Logs

- **Application logs**: `logs/app.log`
- **Container logs**: `docker logs <container-id>`
- **Streamlit logs**: Usually in stdout

### Common Issues

1. **Memory errors**: Increase container memory limits
2. **Model download fails**: Check internet connection, verify model names
3. **Data files missing**: Run preprocessing pipeline
4. **Import errors**: Check requirements.txt, rebuild container

## 📊 Performance Optimization

### For Production

1. **Pre-download models**:
   ```bash
   # In Dockerfile or init script
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
   ```

2. **Optimize data loading**:
   - Use memory mapping for large embeddings
   - Pre-normalize embeddings
   - Cache frequently accessed data

3. **Resource management**:
   - Use slim base images
   - Multi-stage Docker builds
   - Implement connection pooling if needed

### Scaling

1. **Horizontal scaling**: Deploy multiple instances behind load balancer
2. **Caching**: Add Redis for search result caching
3. **CDN**: Serve static assets via CDN
4. **Database**: Move to proper database for large datasets

## 🔐 Security

### Production Security

1. **Container security**:
   ```dockerfile
   # Run as non-root user (already in Dockerfile)
   USER appuser
   ```

2. **Network security**:
   - Use HTTPS in production
   - Implement rate limiting
   - Add firewall rules

3. **Data security**:
   - Encrypt data at rest
   - Use secrets management for sensitive config
   - Regular security updates

### Environment Security

```bash
# Production .env example
ENVIRONMENT=production
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
HTTPS_ONLY=true
```

## 🚨 Troubleshooting

### Quick Diagnostics

```bash
# Check all systems
python health_check.py

# Test basic functionality
python run_tests.py

# Check data files
ls -la data/processed/

# Check container health
docker ps
docker logs <container-name>
```

### Common Solutions

1. **App won't start**: Check data files exist, run health_check.py
2. **Slow performance**: Check memory limits, optimize model loading
3. **Search not working**: Verify embeddings are normalized, check BM25 index
4. **Models not downloading**: Check internet, verify model names in config

## 📞 Getting Help

If you encounter issues:

1. Check logs for error messages
2. Run diagnostic scripts
3. Verify all prerequisites are met
4. Check GitHub issues for similar problems