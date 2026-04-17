@echo off
echo Starting Renewal Intelligence Engine...

:: Install any missing requirements
pip install -r requirements.txt >nul 2>&1

:: Run the streamlit app
echo Launching Dashboard (will use LLM key from .env if present)
streamlit run app.py
