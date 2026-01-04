#!/usr/bin/env python
"""
Generate Figure 2 from the manuscript: policy feasibility map.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analysis import compute_policy_feasibility
from src.plot_utils import plot_policy_feasibility

def main():
    # Compute feasibility data
    feasibility_data = compute_policy_feasibility(
        c_policy=1.0, 
        b_range=(0.0, 1.0), 
        gamma_range=(0.0, 1.0), 
        p_star=0.7,
        n_points=100
    )
    
    # Generate figure
    fig, ax = plot_policy_feasibility(feasibility_data)
    
    # Save figure
    output_dir = "../figures"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "figure2_policy_feasibility.png"), 
                dpi=300, bbox_inches='tight')
    
    print(f"Figure 2 saved to {output_dir}/figure2_policy_feasibility.png")

if __name__ == "__main__":
    main()