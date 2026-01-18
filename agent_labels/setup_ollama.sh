#!/bin/bash
# Setup Ollama for local agent labeling
#
# This script installs Ollama and pulls the llava vision model.
# Run this once before using: ./run_agent.sh ollama
#
# Note: This task requires at least a 7B model. Smaller models like
# moondream (1.8B) struggle with the spatial reasoning required.

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
    ollama serve >/dev/null 2>&1 &

    # Wait for server with retry loop (more reliable than fixed sleep)
    echo "Waiting for Ollama server..."
    max_attempts=30
    attempt=1
    until curl -s http://localhost:11434/api/tags &>/dev/null 2>&1; do
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "Error: Ollama server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo "Ollama server started."
    echo ""
fi

# Pull recommended vision model
echo "Pulling llava (7B vision model, ~4GB download)..."
ollama pull llava

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  ./run_agent.sh ollama --model llava"
echo ""
echo "Other vision models:"
echo "  ollama pull llava:13b    # LLaVA 13B (better quality, needs more VRAM)"
echo ""
echo "Note: Requires GPU. CPU-only inference is too slow for this task."
echo ""
