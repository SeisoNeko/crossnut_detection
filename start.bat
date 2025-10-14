@echo off
chcp 65001 >nul
title 道路挖掘偵測系統

echo.
echo ========================================
echo 道路挖掘偵測系統 - Windows 啟動器
echo ========================================
echo.

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

.conda\python.exe main.py %device%

if errorlevel 1 (
	echo.
	echo ⚠️ 發生錯誤！請聯絡開發人員！
) else (
	echo.
	echo 👋 已完成！將開啟output資料夾！
	pause
	explorer .\output
)

