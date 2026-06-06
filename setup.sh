#!/bin/bash
# setup.sh - Create virtual environment and install dependencies for mincli

set -e  # Exit immediately on error

echo "🔧 Creating Python virtual environment..."

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found, please install Python 3.8+."
    exit 1
fi

# Create virtual environment (skip if exists)
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created: venv/"
else
    echo "⚠️  Virtual environment already exists, skipping."
fi

# Use venv/bin/python and venv/bin/pip directly
echo "📦 Installing dependencies..."

venv/bin/pip install --upgrade pip
venv/bin/pip install \
    tiktoken \
    typer \
    "python-dotenv>=1.0.0" \
    "openai>=1.0.0" \
    rich \
    prompt-toolkit \
    pdfminer.six \
    python-docx \
    trafilatura

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "👉 Activate virtual environment with:"
echo "   source venv/bin/activate"
echo ""
echo "👉 Run mincli:"
echo "   python main.py chat"