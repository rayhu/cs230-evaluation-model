#!/bin/bash

# CS230 Evaluation Model - Quick Setup Script

set -e  # Exit on error

echo "🚀 Starting CS230 project environment setup..."
echo ""

# Check Python version
echo "📋 Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Found Python: $PYTHON_VERSION"
else
    echo "❌ Error: Python 3 not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created successfully"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "📥 Installing project dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✨ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "1. Activate virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run download script:"
echo "   python scripts/download_pubtables_raw.py"
echo ""
echo "3. View setup guide:"
echo "   cat SETUP.md"
echo ""
echo "🎉 Ready to start coding!"
