@echo off
call cuda_env\Scripts\activate.bat
pytest tests\ -v
pause
