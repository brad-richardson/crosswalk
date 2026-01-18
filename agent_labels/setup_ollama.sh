#!/bin/bash
# Setup Ollama for local agent labeling
#
# This script installs Ollama and pulls the moondream vision model.
# Run this once before using: ./run_agent.sh ollama

set -e

echo "=== Ollama Setup for Agent Labeling ==="
echo ""

# Check if Ollama is installed
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo ""
fi

# Check if jq is installed (required for JSON parsing)
if ! command -v jq &>/dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install with: sudo apt install jq"
    exit 1
fi

# Start Ollama server if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3

    # Verify server started
    if ! curl -s http://localhost:11434/api/tags &>/dev/null 2>&1; then
        echo "Error: Failed to start Ollama server"
        exit 1
    fi
    echo "Ollama server started."
    echo ""
fi

# Pull recommended vision model
echo "Pulling moondream (1.8B vision model, ~1GB download)..."
ollama pull moondream

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  ./run_agent.sh ollama --model moondream       # GPU mode (faster)"
echo "  ./run_agent.sh ollama --model moondream --cpu # CPU mode (no GPU)"
echo ""
echo "Other vision models you can try:"
echo "  ollama pull llava        # LLaVA 7B (better quality, slower)"
echo "  ollama pull llava:13b    # LLaVA 13B (best quality, slowest)"
echo ""
