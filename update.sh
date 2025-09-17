#!/usr/bin/env bash

echo "Navigating to parent directory..."
cd ..

echo "Pulling latest changes for the main project..."
git pull origin main

echo "Updating submodules..."
git submodule update --init --recursive

echo ""
echo "✅ Update complete!"

# --- NEW LINES ADDED BELOW ---
echo "↩️  Returning to worldmem directory..."
cd worldmem
