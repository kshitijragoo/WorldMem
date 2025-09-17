#!/usr/bin/env bash

echo "Navigating to parent directory..."
cd ..

echo "Pulling latest changes for the main project..."
git pull origin main

echo "Force cleaning submodules to remove local changes and untracked files..."
git submodule foreach 'git reset --hard && git clean -fde *.npz -e *.pth -e *.bin -e *.ckpt'

echo "Updating submodules..."
git submodule update --init --recursive

echo ""
echo "✅ Update complete!"

# --- NEW LINES ADDED BELOW ---
echo "↩️  Returning to worldmem directory..."
cd worldmem
