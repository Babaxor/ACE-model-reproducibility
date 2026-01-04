#!/usr/bin/env python
"""
Generate Figure 1 from the manuscript: stability dependence on c.
"""

import sys
import os

# Add parent directory to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ode_solver import ACEModel
from src.plot_utils import plot_stability_interval

def main():
    # Create model with baseline parameters
    model = ACEModel(a_tilde=1.0, b=0.2, gamma=0.5, c=1.0, p_star=0.7)
    
    # Generate figure
    fig, ax = plot_stability_interval(model, c_range=(0, 4), n_points=1000)
    
    # Save figure
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "figure1_stability_vs_c.png"), 
                dpi=300, bbox_inches='tight')
    
    print(f"Figure 1 saved to {output_dir}/figure1_stability_vs_c.png")
    print(f"L = {model.L:.3f}, U = {model.U:.3f}, c = {model.c}")

if __name__ == "__main__":
    main()