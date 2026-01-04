#!/usr/bin/env python
"""
Compute separatrices for different c values.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ode_solver import ACEModel
from src.separatrix import compute_separatrix_curve, save_separatrix_data

def main():
    # We'll compute separatrices for three cases: c below L, between L and U, and above U
    baseline_params = {
        'a_tilde': 1.0,
        'b': 0.2,
        'gamma': 0.5,
        'p_star': 0.7,
        'r': 1.0,
        'lam': 0.3,
        'K': 1.0
    }
    
    # Create model to compute L and U
    model_for_bounds = ACEModel(**baseline_params, c=1.0)  # c arbitrary for bounds
    L = model_for_bounds.L
    U = model_for_bounds.U
    
    print(f"L = {L:.3f}, U = {U:.3f}")
    
    # Three c values
    c_values = [L * 0.8, (L + U) / 2, U * 1.2]
    c_labels = ['c_below_L', 'c_between', 'c_above_U']
    
    for c_val, label in zip(c_values, c_labels):
        print(f"\nComputing separatrix for c = {c_val:.3f} ({label})...")
        
        # Create model with this c
        model = ACEModel(**baseline_params, c=c_val)
        
        # Compute separatrix
        separatrix_data = compute_separatrix_curve(
            model, 
            xT_values=None,  # Use default log-spaced values
            V_min=0.0,
            V_max=2.0,
            epsilon=1e-4,
            max_iter=50,
            t_max=200
        )
        
        # Save data
        output_dir = "../data/separatrices"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"separatrix_{label}.csv")
        save_separatrix_data(separatrix_data, filename)
        
        print(f"  Saved to {filename}")
    
    print("\nSeparatrix computation complete.")

if __name__ == "__main__":
    main()