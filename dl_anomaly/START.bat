@echo off
chcp 65001 >nul
title DL Anomaly Detector - Industrial Vision
cd /d "%~dp0"
"C:\Users\User\anaconda3\envs\cnn-transformer\python.exe" main.py
if errorlevel 1 pause
