@echo off
if not exist data mkdir data
py download_data.py
if errorlevel 1 pause
py -m streamlit run app.py
pause
