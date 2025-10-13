@echo off
chcp 65001 >nul
title 道路挖掘偵測系統

echo.
echo ========================================
echo 道路挖掘偵測系統 - Windows 啟動器
echo ========================================
echo.

REM 檢查 Python 是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤：未找到 Python
    echo    請先安裝 Python 3.8 或更高版本
    echo    下載地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python 已安裝

REM 檢查 Cuda 是否安裝
nvidia-smi >nul 2>&1
if errorlevel 1 (
	echo ⚠️ 提示：未找到 Cuda，將使用 CPU 模式
	set device="cpu"
	goto START
)

REM 檢查 CuDNN 是否安裝
nvcc -V >nul 2>&1
if errorlevel 1 (
	echo ⚠️ 提示：未找到 CuDNN，將使用 CPU 模式
	set device="cpu"
	goto START
) else (
	set device="cuda"
)

echo ✅ CuDNN 已安裝

:START
echo.
REM 啟動系統
echo 🚀 啟動系統...

python main.py %device%

echo.
echo 👋 系統已停止
pause
