#!/bin/bash
# verify.sh - Reproduce all results and figures from the manuscript

echo "========================================"
echo "Reproducing ACE model results"
echo "========================================"

# Create output directories
mkdir -p figures
mkdir -p figures/sensitivity
mkdir -p data/outputs
mkdir -p data/separatrices
mkdir -p data/sensitivity

echo "1. Running symbolic derivations..."
cd scripts
python run_notebooks.py
cd ..

echo "2. Computing stability bounds..."
python -c "
from src.ode_solver import ACEModel
model = ACEModel()
print(f'L = {model.L:.3f}, U = {model.U:.3f}')
print(f'c = {model.c} is stable: {model.check_stability()[0]}')
"

echo "3. Generating Figure 1 (stability vs c)..."
python scripts/generate_figure1.py

echo "4. Generating Figure 2 (policy feasibility)..."
python scripts/generate_figure2.py

echo "5. Running sensitivity analysis..."
python scripts/run_sensitivity.py

echo "6. Computing separatrices..."
python scripts/compute_separatrices.py

echo "========================================"
echo "Reproduction complete!"
echo "Results saved in:"
echo "  - figures/"
echo "  - data/"
echo "========================================"