# Release Notes

## Version 1.0.0 - Manuscript Release (2024-03-27)

### Overview
This is the initial release accompanying the manuscript:
**"Local ESS Bounds in a Unidirectionally Coupled Replicator-Resource System: The Three-Strategy ACE Reduction"**

### New Features
- **Complete reproducibility** of all results in the manuscript
- **Modular Python implementation** with clean separation of concerns
- **Symbolic derivations** in Jupyter notebooks (SymPy)
- **Command-line interface** for batch processing
- **Comprehensive test suite** with high code coverage
- **Automated CI/CD pipeline** via GitHub Actions
- **Publication-quality figures** generation
- **Full documentation** with usage examples

### Components

#### Core Modules (`src/`)
- `model.py`: ACE model definition and parameter management
- `compute_bounds.py`: Analytical stability bounds (Theorem 3.3)
- `ode_solver.py`: Numerical integration of coupled ODEs
- `separatrix.py`: Basin of attraction analysis
- `analysis.py`: Sensitivity and feasibility analysis
- `plot_utils.py`: Visualization utilities

#### Notebooks (`notebooks/`)
- `01_LU_derivation.ipynb`: Symbolic derivation of L and U bounds
- `02_Jacobian_analysis.ipynb`: Jacobian diagonalization proof
- `03_asymmetric_verification.ipynb`: Extended analysis
- `99_reproduce_figures.ipynb`: Complete reproduction pipeline

#### Scripts (`scripts/`)
- `generate_figure1.py`: Generate Figure 1 (stability vs c)
- `generate_figure2.py`: Generate Figure 2 (policy feasibility)
- `run_sensitivity.py`: Sensitivity analysis (Section 5.3)
- `compute_separatrices.py`: Separatrix computation
- `compute_bounds_cli.py`: Command-line interface
- `run_notebooks.py`: Notebook execution wrapper

#### Tests (`tests/`)
- `test_compute_bounds.py`: Unit tests for analytical results
- `test_jacobian.py`: Tests for Jacobian properties
- `test_ode_solver.py`: Tests for numerical integration

### System Requirements
- Python 3.9 or higher
- Required packages: numpy, scipy, sympy, matplotlib, pandas
- See `requirements.txt` for complete list

### Installation
```bash
git clone https://github.com/username/ACE-model-reproducibility.git
cd ACE-model-reproducibility
pip install -r requirements.txt