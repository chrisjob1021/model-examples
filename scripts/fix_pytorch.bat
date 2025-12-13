@echo off
echo Fixing PyTorch for RTX 5090 compatibility...

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Uninstall current PyTorch
echo Uninstalling current PyTorch...
pip uninstall torch torchvision torchaudio -y

REM Install latest PyTorch with CUDA 12.4 support
echo Installing latest PyTorch with CUDA 12.4 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Verify installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}')"

echo.
echo PyTorch fix complete!
pause 