#!/bin/bash

echo "🧬 PROTEUS Backend - Local Development"
echo "======================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1

echo ""
echo "✅ Environment ready!"
echo ""
echo "Starting PROTEUS backend server..."
echo "WebSocket: ws://localhost:8000/ws/{client_id}"
echo "Stats: http://localhost:8000/stats"
echo ""

# Run the server
python run_server.py