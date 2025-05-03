#!/bin/bash
# Helper script to build Docker images with uv lock file

# Check for data directory structure
mkdir -p data/raw data/processed data/logs data/recommendations data/checkpoints models

# Generate uv.lock file if it doesn't exist
if [ ! -f "uv.lock" ]; then
    echo "Generating uv.lock file from requirements.txt..."
    uv pip compile requirements.txt --output-file uv.lock
fi

# Build the Docker images
docker-compose build

echo "Docker images built successfully!"