#!/bin/bash

# Script to push backend API code to GitHub repository
# Repository: https://github.com/Yakesh70/cvvrs-backend-.git

echo "🚀 Pushing backend API code to GitHub..."
echo "Repository: https://github.com/Yakesh70/cvvrs-backend-.git"
echo ""

cd "$(dirname "$0")"

# Update remote URL
echo "📝 Updating remote URL..."
git remote set-url origin https://github.com/Yakesh70/cvvrs-backend-.git

# Verify remote
echo "✅ Remote configured:"
git remote -v
echo ""

# Check status
echo "📋 Current status:"
git status
echo ""

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push -u origin main --force

echo ""
echo "✅ Done! Your backend code is now on GitHub."
echo "🌐 Next step: Go to https://railway.app and deploy from this repository"
