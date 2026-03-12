#!/bin/bash
# Run the PolicyCapture Local server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Ensure data directory exists
mkdir -p data/jobs

echo "Starting PolicyCapture Local server..."
echo "Dashboard: http://localhost:8420"
echo ""

python -m uvicorn apps.local_api.main:app --host 0.0.0.0 --port 8420 --reload
