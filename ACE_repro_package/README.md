# Reproducibility Package for "Local ESS Bounds in a Unidirectionally Coupled Replicator-Resource System"

This repository contains all code, data, and scripts to reproduce the results in the paper.

## Structure

- `src/`: Python modules implementing the ACE model, ODE solver, analysis, and plotting.
- `notebooks/`: Jupyter notebooks for symbolic derivations.
- `scripts/`: Python scripts to generate figures and run analyses.
- `figures/`: Generated figures (created when scripts are run).
- `data/`: Output data from simulations (created when scripts are run).

## Requirements

See `requirements.txt` for Python package dependencies. To install:

```bash
pip install -r requirements.txt