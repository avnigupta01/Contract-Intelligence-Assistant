@echo off
setlocal
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_data.py
python -m src.validate_data
python -m pytest -q
echo.
echo Setup finished. Start the web app with: streamlit run app.py
pause
