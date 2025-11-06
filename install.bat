@echo off
echo ========================================
echo  Canteen Menu Optimizer - Quick Setup
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Installing backend dependencies...
cd canteen-App\backend
pip install -r requirements.txt
cd ..\..

echo.
echo Installing frontend dependencies...
cd canteen-App\frontend
pip install -r requirements.txt
cd ..\..

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo To start the application:
echo 1. Backend:  cd canteen-App\backend ^&^& python main.py
echo 2. Frontend: cd canteen-App\frontend ^&^& streamlit run app.py
echo 3. Open: http://localhost:8501
echo.
pause