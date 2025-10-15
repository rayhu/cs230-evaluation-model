#!/bin/bash

# Start Jupyter Lab for CS230 Evaluation Model

set -e  # Exit on error

echo "🚀 Starting Jupyter Lab..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "Please run ./setup.sh first to create the environment"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Check if JupyterLab is installed
if ! command -v jupyter-lab &> /dev/null; then
    echo "📥 Installing matplotlib and JupyterLab..."
    pip install matplotlib jupyterlab --quiet
fi

# Start Jupyter Lab
echo "✅ Launching Jupyter Lab..."
echo ""
echo "📝 Jupyter Lab will open in your browser automatically"
echo "   If not, click the URL shown below"
echo ""
echo "🛑 To stop: Press Ctrl+C"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

jupyter lab --notebook-dir=/Users/rayhu/play/ai/cs230-evaluation-model

