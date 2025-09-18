# git_upload_steps.md - Simple Git Upload Guide

## Step 1: Local Setup (On Your Computer)

# Navigate to your project folder
cd path\to\your\flux-runpod-project

# Initialize git (if not already done)
git init

# Set your git info (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "243bairyk@gmail.com"

## Step 2: Create .gitignore

# Create .gitignore file to exclude unnecessary files
echo "*.pyc" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.log" >> .gitignore
echo ".env" >> .gitignore
echo "logs/" >> .gitignore
echo "outputs/" >> .gitignore

## Step 3: Add Files to Git

# Add all files to git
git add .

# Create first commit
git commit -m "Initial FLUX RunPod setup"

## Step 4: Create GitHub Repository

# Go to github.com in browser
# Click "New repository"  
# Name it "flux-runpod-api"
# Make it public
# Don't initialize with README
# Copy the repository URL

## Step 5: Connect and Push to GitHub

# Connect to your GitHub repo (replace with your URL)
git remote add origin https://github.com/yourusername/flux-runpod-api.git

# Push to GitHub
git push -u origin main

## Step 6: Clone on RunPod

# SSH into your RunPod
ssh 8tyra45f9q5zle-64411644@ssh.runpod.io -i ~/.ssh/id_ed25519

# Navigate to workspace
cd /workspace

# Clone your repository
git clone https://github.com/yourusername/flux-runpod-api.git

# Enter project directory  
cd flux-runpod-api

# Run setup
chmod +x setup.sh
./setup.sh

## Quick Commands Summary:

# On your computer:
git add .
git commit -m "Update files"
git push

# On RunPod:
git pull  # to get latest changes

# That's it!
