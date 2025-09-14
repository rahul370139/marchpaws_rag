#!/bin/bash

# Enhanced MARCH-PAWS Medical Assistant Launcher
echo "🏥 Starting Enhanced MARCH-PAWS Medical Assistant..."
echo "🚀 Activating virtual environment..."
source venv/bin/activate

echo "📦 Checking dependencies..."
python3 -c "
import sys
sys.path.append('src')
try:
    from orchestrator_async import AsyncOrchestrator
    from quality_evaluator import QualityEvaluator
    print('✅ All dependencies ready!')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
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
