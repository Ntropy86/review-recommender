#!/bin/bash
# start.sh - Production startup script

set -e

echo "🚀 Starting Review Search Copilot..."

# Set default environment
export ENVIRONMENT=${ENVIRONMENT:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Validate configuration
echo "📋 Validating configuration..."
python -c "from config import config; config.validate(); print('✅ Configuration valid')"

# Check data files
echo "📁 Checking data files..."
python -c "
from config import config
import sys
critical_files = [config.PRODUCT_EMB_PATH, config.PRODUCT_META_PATH]
missing = [f for f in critical_files if not f.exists()]
if missing:
    print(f'❌ Critical files missing: {missing}')
    sys.exit(1)
print('✅ All critical data files present')
"

# Run tests if in development
if [ "$ENVIRONMENT" = "development" ]; then
    echo "🧪 Running tests..."
    python run_tests.py
fi

# Start the application
echo "🌟 Starting Streamlit application..."
exec streamlit run app/app_product_search.py \
    --server.port="${APP_PORT:-8501}" \
    --server.address="${APP_HOST:-0.0.0.0}" \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false \
    --logger.level="${LOG_LEVEL:-INFO}"