@echo off
setlocal

REM ===============================
REM Activate conda env (EDIT NAME)
REM ===============================
call "%USERPROFILE%\miniforge3\Scripts\activate.bat" david-env

REM ===============================
REM Shared base arguments
REM ===============================
set BASE=python capture_correlation_functions.py ^
 --k 3 ^
 --im_width 512 ^
 --bit_depth 12 ^
 --int_time 200 ^
 --burst_time 480 ^
 --shift 2500 ^
 --rep_rate 5000000 ^
 --current 75 ^
 --amplitude 2.0 ^
 --plot_correlations false ^
 --save_into_file true


REM ===============================
REM Run 1 (coarse)
REM ===============================
%BASE% ^
 --gate_shrinkage 10 ^
 --capture_type coarse ^
 --duty 32 ^
 --illum_type gaussian

timeout /t 5 >nul


REM ===============================
REM Run 2 (ham)
REM ===============================
%BASE% ^
 --gate_shrinkage 25 ^
 --capture_type ham ^
 --duty 25 ^
 --illum_type square

timeout /t 5 >nul


pause

