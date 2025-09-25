#!/usr/bin/env bash

echo "Navigating to parent directory..."
cd ..

# --- MODIFIED SECTION ---
# Dynamically get the name of the current branch (e.g., "main", "branch1")
current_branch=$(git rev-parse --abbrev-ref HEAD)

echo "Fetching all latest changes from the remote repository..."
git fetch origin

echo "Pulling latest changes for current branch: '$current_branch'..."
git pull origin "$current_branch"

# Only merge 'main' if the current branch is not 'main'
if [ "$current_branch" != "main" ]; then
    echo "Merging latest changes from 'main' into '$current_branch'..."
    git merge origin/main
else
    echo "Current branch is 'main', no separate merge needed."
fi
# --- END MODIFIED SECTION ---

echo "Force cleaning submodules to remove local changes and untracked files..."
git submodule foreach 'git reset --hard && git clean -fde *.npz -e *.pth -e *.bin -e *.ckpt'

echo "Updating submodules to match the new state of '$current_branch'..."
git submodule update --init --recursive

echo ""
echo "✅ Update complete!"

# --- NEW LINES ADDED BELOW ---
echo "↩️  Returning to worldmem directory..."
cd worldmem

git checkout memory_viz
git pull