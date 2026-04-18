#!/bin/bash
# Rakshak GitHub Push Script

echo "🚀 Rakshak GitHub Push Script"
echo "================================"
echo ""

# Configuration
APP_DIR="/Workspace/Users/mc230041001@iiti.ac.in/databricks_apps/digital-artha-sarvam_2026_04_18-07_08/streamlit-hello-world-app"
REPO_NAME="databricks-hackathon"

echo "📁 App Directory: $APP_DIR"
echo "🔗 Target Repo: $REPO_NAME"
echo ""

# Navigate to app directory
cd "$APP_DIR"

# Initialize git (if not already initialized)
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Configure git
echo ""
echo "📝 Git Configuration"
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter your name: " USER_NAME
read -p "Enter your email: " USER_EMAIL
read -sp "Enter your GitHub Personal Access Token: " GITHUB_TOKEN
echo ""

git config user.name "$USER_NAME"
git config user.email "$USER_EMAIL"
echo "✅ Git config set"

# Add all files
echo ""
echo "📦 Adding files to Git..."
git add .
echo "✅ Files added"

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short

# Commit
echo ""
read -p "Press Enter to commit..."
git commit -m "Add Rakshak Financial Intelligence Platform

- Fraud Detection with XGBoost + DQN hybrid model
- Credit Eligibility assessment (banking behavior)
- Multilingual AI Assistant with Sarvam-1 RAG
- Support for 10+ Indian languages (Hindi, Tamil, Telugu, Bengali, etc.)
- Built on Databricks Apps with Streamlit
- CPU-optimized with 8-bit quantization
- Complete UI for all features"

echo "✅ Committed"

# Add remote
echo ""
echo "🔗 Setting up remote repository..."
git remote add origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
echo "✅ Remote added"

# Push to GitHub
echo ""
echo "⬆️  Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "🎉 SUCCESS! Your code is now on GitHub!"
echo "🔗 Visit: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
