@echo off
cd /d "%~dp0"
title Local PDF Extractor
echo Starting Local PDF Extractor...
echo (You can minimize this window. Closing it stops the app.)
".venv\Scripts\python.exe" desktop.py
