#!/bin/bash

# Enhanced MARCH-PAWS Medical Assistant Launcher
echo "🏥 Starting Enhanced MARCH-PAWS Medical Assistant..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

echo "🚀 Activating virtual environment..."
source venv/bin/activate

echo "📦 Checking and installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements"
        exit 1
    fi
    echo "✅ Requirements installed successfully"
else
    echo "⚠️  requirements.txt not found, skipping installation"
fi

echo "🔍 Verifying dependencies..."
python3 -c "
import sys
sys.path.append('src')
try:
    from orchestrator_async import AsyncOrchestrator
    from quality_evaluator import QualityEvaluator
    print('✅ All dependencies ready!')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Please install missing dependencies manually')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "🌐 Starting Streamlit app..."
    echo "📍 Open your browser to: http://localhost:8501"
    echo ""
    streamlit run app.py
else
    echo "❌ Please install missing dependencies first"
    exit 1
fi
