@echo off
REM ============================================================================
REM  This batch script runs the split_CAP.py script for a predefined list
REM  of subjects from the CAP Sleep Database.
REM ============================================================================

REM --- Configuration ---
REM Set the path to your Python executable
SET PYTHON_EXE="C:/Users/klanc/miniforge3/envs/Magistrska/python.exe"

REM Set the path to your Python script
SET SCRIPT_PATH="c:/My Things/PV2/masters-sleep-stage-detection/Prepare data/split_CAP.py"

REM --- Execution ---
echo Starting script execution...

REM Category: No pathology (controls) (n1-n16)
echo.
echo Processing No pathology (controls)...
FOR /L %%i IN (1, 1, 16) DO (
    echo Running for n%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "n%%i"
)

REM Category: Bruxism (brux1-brux2)
echo.
echo Processing Bruxism...
FOR /L %%i IN (1, 1, 2) DO (
    echo Running for brux%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "brux%%i"
)

REM Category: Insomnia (ins1-ins9)
echo.
echo Processing Insomnia...
FOR /L %%i IN (1, 1, 9) DO (
    echo Running for ins%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "ins%%i"
)

REM Category: Narcolepsy (narco1-narco5)
echo.
echo Processing Narcolepsy...
FOR /L %%i IN (1, 1, 5) DO (
    echo Running for narco%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "narco%%i"
)

REM Category: Nocturnal frontal lobe epilepsy (nfle1-nfle40)
echo.
echo Processing Nocturnal frontal lobe epilepsy...
FOR /L %%i IN (1, 1, 40) DO (
    echo Running for nfle%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "nfle%%i"
)

REM Category: Periodic leg movements (plm1-plm10)
echo.
echo Processing Periodic leg movements...
FOR /L %%i IN (1, 1, 10) DO (
    echo Running for plm%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "plm%%i"
)

REM Category: REM behavior disorder (rbd1-rbd22)
echo.
echo Processing REM behavior disorder...
FOR /L %%i IN (1, 1, 22) DO (
    echo Running for rbd%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "rbd%%i"
)

REM Category: Sleep-disordered breathing (sdb1-sdb4)
echo.
echo Processing Sleep-disordered breathing...
FOR /L %%i IN (1, 1, 4) DO (
    echo Running for sdb%%i
    %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "sdb%%i"
)

echo.
echo All tasks completed.
pause