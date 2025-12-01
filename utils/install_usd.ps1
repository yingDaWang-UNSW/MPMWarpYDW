# USD Renderer Installation Script
# Run this to install USD Python bindings for GPU-direct rendering

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "USD Renderer Setup for MPM Simulation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor Green

# Try to install USD via pip
Write-Host ""
Write-Host "Installing USD Python bindings..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray

try {
    pip install usd-core --upgrade
    Write-Host "  ✓ USD bindings installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Failed to install via pip" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative installation methods:" -ForegroundColor Yellow
    Write-Host "  1. Conda: conda install -c conda-forge usd-py" -ForegroundColor Gray
    Write-Host "  2. Build from source: https://github.com/PixarAnimationStudios/USD" -ForegroundColor Gray
    Write-Host "  3. Use fallback mode (automatic, no installation needed)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "The renderer will work in fallback mode without USD bindings," -ForegroundColor Cyan
    Write-Host "but performance will be reduced." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test USD rendering, run:" -ForegroundColor Yellow
Write-Host "  python runMPMYDW.py --config benchmarks/generalBenchmark/config_quick_test.json" -ForegroundColor Gray
Write-Host ""
Write-Host "Output will be saved to:" -ForegroundColor Yellow
Write-Host "  ./benchmarks/generalBenchmark/outputElastic/usd/simulation.usd" -ForegroundColor Gray
Write-Host ""
Write-Host "View with:" -ForegroundColor Yellow
Write-Host "  usdview ./benchmarks/generalBenchmark/outputElastic/usd/simulation.usd" -ForegroundColor Gray
Write-Host "  (or open in NVIDIA Omniverse)" -ForegroundColor Gray
Write-Host ""
