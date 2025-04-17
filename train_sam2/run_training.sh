#!/bin/bash
# SAM2 Training wrapper script
# This script sets up the environment and runs the SAM2 training pipeline

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Using existing virtual environment..."
    source venv/bin/activate
fi

# Check for API key in .env file
if [ ! -f "../.env" ]; then
    echo "Warning: .env file not found in parent directory!"
    echo "Make sure your API key is properly set before training."
fi

# Run the training pipeline
python run_sam2_training.py "$@"

# Deactivate virtual environment
deactivate 