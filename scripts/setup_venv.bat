@echo off
echo Setting up Python virtual environment...

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete! Virtual environment is active.
echo To activate in the future: .venv\Scripts\activate.bat
echo To deactivate: deactivate
pause 