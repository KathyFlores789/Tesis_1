@echo off
title Servidor de Mantenimiento Predictivo - Tesis MIA
echo ===================================================
echo Iniciando Backend (FastAPI) y Abriendo Dashboard...
echo ===================================================

:: Cambiar a la carpeta actual donde se ubica este archivo bat
cd /d "%~dp0"

:: Activar el entorno virtual de Python
call "%~dp0.venv\Scripts\activate.bat"

:: Iniciar la API con Uvicorn en segundo plano
start /b python -m uvicorn src.api:app --reload

:: Esperar 3 segundos para asegurar que la API encienda
timeout /t 3 /nobreak > nul

:: Abrir el Dashboard HTML directamente en el navegador por defecto
start "" "%~dp0dashboard.html"

echo.
echo ===================================================
echo   SISTEMA ACTIVO Y CONECTADO 🟢
echo   Puedes minimizar esta ventana.
echo   Para cerrar el servidor presiona Ctrl + C o cierra esta ventana.
echo ===================================================
echo.

:: Mantener la consola abierta escuchando peticiones
cmd /k