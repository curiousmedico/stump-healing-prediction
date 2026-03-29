#!/bin/bash
# Setup script for creating Python virtual environment for Stump Healing Research

echo "========================================="
echo "Setting up Python Virtual Environment"
echo "========================================="

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo ""
echo "✓ Virtual environment created and activated!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Ready to run: python shap_integration.py"
