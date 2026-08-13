#!/bin/bash
# Activate the primary virtual environment for the UI
source .venv_mac/bin/activate

# Start the Streamlit application
echo "Starting Wall Crack Detection Analytics Dashboard..."
streamlit run app.py
