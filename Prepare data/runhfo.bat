@echo off
REM ============================================================================
REM  This batch script runs the split_CAP.py script for a predefined list
REM  of subjects from the CAP Sleep Database.
REM ============================================================================

REM --- Configuration ---
REM Set the path to your Python executable
SET PYTHON_EXE="C:/Users/klanc/miniforge3/envs/Magistrska/python.exe"

REM Set the path to your Python script
SET SCRIPT_PATH="c:/My Things/PV2/masters-sleep-stage-detection/Prepare data/split_HFO.py"

REM Enable delayed variable expansion
SETLOCAL EnableDelayedExpansion

REM --- Execution ---
echo Starting script execution...

REM Category: No pathology (controls) (n1-n16)
echo.
echo Processing No pathology (controls)...
FOR /L %%i IN (1, 1, 30) DO (
    SET "num=%%i"
    IF %%i LSS 10 (
        SET "num=0%%i"
    )
    echo Running for !num!
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "!num!"
)



echo.
echo All tasks completed.
pause