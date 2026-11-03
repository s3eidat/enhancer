@echo off
rem Starts S3eidat Enhancer from this source folder (needs .venv and models/, see README).
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m enhancer
