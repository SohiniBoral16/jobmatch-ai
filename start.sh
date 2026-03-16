#!/bin/bash
# JobMatch AI — Start Script
# Run from the project root: bash start.sh

cd "$(dirname "$0")"

echo "🎯 Starting JobMatch AI..."
echo "   Open your browser at: http://localhost:8080"
echo ""

python3 -m streamlit run app.py \
  --server.port 8080 \
  --server.address localhost \
  --browser.gatherUsageStats false
