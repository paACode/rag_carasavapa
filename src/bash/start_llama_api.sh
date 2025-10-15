#!/bin/bash

# Please Note: Make sure to install Ollama first
# - Windows/Mac/Linux: Download from https://ollama.com/download

# Run the following commands in bash

# Download Models to local Machine
ollama pull llama3.2    # For Testing
ollama pull mistral:7b  # Should e good in reasoning

# Start Ollama in server mode for API access
echo "Starting Ollama API server on http://localhost:11434 ..."
ollama serve