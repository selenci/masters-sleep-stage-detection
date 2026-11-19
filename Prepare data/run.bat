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

@REM REM Category: Bruxism (brux1-brux2)
@REM echo.
@REM echo Processing Bruxism...
@REM FOR /L %%i IN (1, 1, 2) DO (
@REM     echo Running for brux%%i
@REM     %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "brux%%i"
@REM )

@REM REM Category: Insomnia (ins1-ins9)
@REM echo.
@REM echo Processing Insomnia...
@REM FOR /L %%i IN (1, 1, 9) DO (
@REM     echo Running for ins%%i
@REM     %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "ins%%i"
@REM )

@REM REM Category: Narcolepsy (narco1-narco5)
@REM echo.
@REM echo Processing Narcolepsy...
@REM FOR /L %%i IN (1, 1, 5) DO (
@REM     echo Running for narco%%i
@REM     %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "narco%%i"
@REM )

@REM REM Category: Nocturnal frontal lobe epilepsy (nfle1-nfle40)
@REM echo.
@REM echo Processing Nocturnal frontal lobe epilepsy...
@REM FOR /L %%i IN (1, 1, 40) DO (
@REM     echo Running for nfle%%i
@REM     %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "nfle%%i"
@REM )

@REM REM Category: Periodic leg movements (plm1-plm10)
@REM echo.
@REM echo Processing Periodic leg movements...
@REM FOR /L %%i IN (1, 1, 10) DO (
@REM     echo Running for plm%%i
@REM     %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "plm%%i"
@REM )

@REM REM Category: REM behavior disorder (rbd1-rbd22)
@REM echo.
@REM echo Processing REM behavior disorder...
@REM FOR /L %%i IN (1, 1, 22) DO (
@REM     echo Running for rbd%%i
@REM     %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "rbd%%i"
@REM )

@REM REM Category: Sleep-disordered breathing (sdb1-sdb4)
@REM echo.
@REM echo Processing Sleep-disordered breathing...
@REM FOR /L %%i IN (1, 1, 4) DO (
@REM     echo Running for sdb%%i
@REM     %PYTHON_EXE% "-W ignore" %SCRIPT_PATH% "sdb%%i"
@REM )

echo.
echo All tasks completed.
pause