@echo off
echo Running General Benchmark - Quick Test
echo =====================================
echo.
cd ..\..
python runMPMYDW.py --config ./benchmarks/generalBenchmark/config_quick_test.json
echo.
echo =====================================
echo Benchmark complete!
pause
