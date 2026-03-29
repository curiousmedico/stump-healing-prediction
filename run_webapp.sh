#!/bin/bash
# Startup script for Stump Healing Prediction Web App

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "🏥 Stump Healing Prediction System - Web App Launcher"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first:"
    echo "  python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "📥 Checking dependencies..."
pip list | grep -q streamlit || {
    echo "Installing missing dependencies..."
    pip install -q -r requirements.txt
}

# Create data directories if needed
mkdir -p data models outputs

echo ""
echo "🚀 Starting Stump Healing Prediction Web App..."
echo "📍 Opening at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit app
streamlit run src/app.py
