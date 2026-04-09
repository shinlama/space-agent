@echo off
setlocal

set "PYTHON_EXE=%REAL_MODEL_PYTHON%"
if "%PYTHON_EXE%"=="" set "PYTHON_EXE=C:\Users\jiwon\Desktop\temp\space-agent-venv313\Scripts\python.exe"
set "EMBEDDING_MODEL=jhgan/ko-sroberta-multitask"
set "SENTIMENT_MODEL=cringepnh/koelectra-korean-sentiment"

if not exist "%PYTHON_EXE%" (
    echo Python executable not found: "%PYTHON_EXE%"
    echo Set REAL_MODEL_PYTHON or edit this file.
    exit /b 1
)

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

if not exist logs mkdir logs

echo [1/5] Processing cafes 0-449
"%PYTHON_EXE%" scripts\generate_real_model_csvs_fast.py --start-cafe 0 --end-cafe 450 --sentiment-batch-size 16 --encode-batch-size 128 --embedding-model "%EMBEDDING_MODEL%" --sentiment-model "%SENTIMENT_MODEL%" > logs\real_chunk_1.log 2>&1
if errorlevel 1 goto :fail

echo [2/5] Processing cafes 450-899
"%PYTHON_EXE%" scripts\generate_real_model_csvs_fast.py --start-cafe 450 --end-cafe 900 --sentiment-batch-size 16 --encode-batch-size 128 --embedding-model "%EMBEDDING_MODEL%" --sentiment-model "%SENTIMENT_MODEL%" > logs\real_chunk_2.log 2>&1
if errorlevel 1 goto :fail

echo [3/5] Processing cafes 900-1349
"%PYTHON_EXE%" scripts\generate_real_model_csvs_fast.py --start-cafe 900 --end-cafe 1350 --sentiment-batch-size 16 --encode-batch-size 128 --embedding-model "%EMBEDDING_MODEL%" --sentiment-model "%SENTIMENT_MODEL%" > logs\real_chunk_3.log 2>&1
if errorlevel 1 goto :fail

echo [4/5] Processing cafes 1350-1798
"%PYTHON_EXE%" scripts\generate_real_model_csvs_fast.py --start-cafe 1350 --end-cafe 1799 --sentiment-batch-size 16 --encode-batch-size 128 --embedding-model "%EMBEDDING_MODEL%" --sentiment-model "%SENTIMENT_MODEL%" > logs\real_chunk_4.log 2>&1
if errorlevel 1 goto :fail

echo [5/5] Combining chunk outputs
"%PYTHON_EXE%" scripts\combine_real_model_csv_parts.py --suffixes part_0000_0450 part_0450_0900 part_0900_1350 part_1350_1799 > logs\real_chunk_combine.log 2>&1
if errorlevel 1 goto :fail

echo Done. Check the generated *_real.csv files in the project root.
exit /b 0

:fail
echo Failed. Check logs in the logs folder.
exit /b 1
