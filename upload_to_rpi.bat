@echo off
chcp 65001 >nul
:: =====================================================
::  upload_to_rpi.bat  -- Upload code + install libs
::  Run from project folder while on RobotCar WiFi
::  Usage:  .\upload_to_rpi.bat
:: =====================================================

set RPI_USER=yorai
set RPI_IP=10.3.141.1
set RPI_HOME=/home/yorai
set RPI=%RPI_USER%@%RPI_IP%

echo.
echo =====================================================
echo   Uploading code to RPi (%RPI%)
echo =====================================================
echo.

echo [1/6] Creating folders on RPi...
ssh %RPI% "mkdir -p %RPI_HOME%/mks"

echo [2/6] Uploading robot_vision.py...
scp robot_vision.py %RPI%:%RPI_HOME%/robot_vision.py

echo [3/6] Uploading rpi_navigator.py...
scp rpi_navigator.py %RPI%:%RPI_HOME%/rpi_navigator.py

echo [4/6] Uploading rpi_control.py...
scp rpi_control.py %RPI%:%RPI_HOME%/rpi_control.py

echo [5/6] Uploading rpi_code.py...
scp rpi_code.py %RPI%:%RPI_HOME%/rpi_code.py

echo [6/6] Uploading mks/Movments.py...
scp mks\Movments.py %RPI%:%RPI_HOME%/mks/Movments.py

echo.
echo =====================================================
echo   Installing Python libraries on RPi...
echo =====================================================
echo.

echo -- Installing main libraries...
ssh %RPI% "pip install --break-system-packages fastapi uvicorn opencv-python numpy pyserial requests ultralytics timm"

echo.
echo -- Installing PyTorch (this may take several minutes)...
ssh %RPI% "pip install --break-system-packages torch torchvision --extra-index-url https://www.piwheels.org/simple"

echo.
echo =====================================================
echo   Done!
echo.
echo   On RPi run:
echo     Terminal 1:  python3 %RPI_HOME%/robot_vision.py
echo     Terminal 2:  python3 %RPI_HOME%/rpi_navigator.py
echo.
echo   Vision + calibration : http://10.3.141.1:8000
echo   Navigation UI (phone): http://10.3.141.1:8002
echo =====================================================
echo.
pause
