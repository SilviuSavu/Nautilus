#!/bin/bash

# NautilusTrader Engine Build Script
# Sprint 2: Container-in-Container Pattern Implementation

set -e

echo "Building NautilusTrader Engine Image..."

# Build the engine image
docker build -f backend/Dockerfile.engine -t nautilus-engine:latest ./backend

echo "Engine image built successfully: nautilus-engine:latest"

# Verify the image
echo "Verifying engine image..."
docker images nautilus-engine:latest

echo "Build complete! The engine image is ready for dynamic container deployment."