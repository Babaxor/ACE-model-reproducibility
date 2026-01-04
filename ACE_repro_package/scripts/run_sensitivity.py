#!/usr/bin/env python
"""
Run sensitivity analysis for various parameters.
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ode_solver import ACEModel
from src.analysis import sensitivity_analysis
from src.plot_utils import plot_parameter_sensitivity

def main():
    # Create baseline model
    model = ACEModel(a_tilde=1.0, b=0.2, gamma=0.5, c=1.0, p_star=0.7)
    
    # Parameters to analyze
    parameters = ['c', 'b', 'gamma', 'a_tilde', 'p_star']
    
    # Run sensitivity for each parameter
    sensitivity_results = {}
    
    for param in parameters:
        print(f"Running sensitivity analysis for {param}...")
        results = sensitivity_analysis(model, parameter=param, range_pct=0.5, n_points=50)
        sensitivity_results[param] = results
        
        # Plot sensitivity for this parameter
        fig, axes = plot_parameter_sensitivity(results)
        
        # Save figure
        output_dir = "../figures/sensitivity"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"sensitivity_{param}.png"), 
                    dpi=300, bbox_inches='tight')
        
        # Save data as JSON
        data_dir = "../data/sensitivity"
        os.makedirs(data_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON
        json_results = {
            'parameter': results['parameter'],
            'values': results['values'].tolist(),
            'L_values': results['L_values'].tolist(),
            'U_values': results['U_values'].tolist(),
            'stable': results['stable'].tolist(),
            'width': results['width'].tolist()
        }
        
        with open(os.path.join(data_dir, f"sensitivity_{param}.json"), 'w') as f:
            json.dump(json_results, f, indent=2)
    
    print("Sensitivity analysis complete.")

if __name__ == "__main__":
    main()