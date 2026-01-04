"""
Separatrix computation for the ACE model.

Implements the bisection algorithm described in Section 5.2 of the manuscript
to compute basins of attraction boundaries.
"""

import numpy as np
from .ode_solver import ACEModel


def find_separatrix_bisection(model, xT_fixed=0.01, V_min=0.0, V_max=2.0, 
                              epsilon=1e-4, max_iter=50, t_max=200):
    """
    Find the critical initial V that separates Tyrant and Architect basins
    for a fixed initial xT using bisection.
    
    Parameters
    ----------
    model : ACEModel
        ACE model instance
    xT_fixed : float
        Fixed initial Tyrant frequency
    V_min, V_max : float
        Search interval for initial V
    epsilon : float
        Bisection tolerance
    max_iter : int
        Maximum number of bisection iterations
    t_max : float
        Simulation time horizon
        
    Returns
    -------
    V_crit : float
        Critical initial V value (separatrix)
    history : list
        Bisection history [(V_test, outcome), ...]
    """
    
    def run_simulation_and_check(V0):
        """Run simulation and check if Architect recovers."""
        # Initial conditions: small Victim frequency
        y0 = [xT_fixed, 0.01, V0]
        sol = model.simulate(t_span=(0, t_max), y0=y0)
        
        # Final Architect frequency
        xT_final, xV_final, V_final = sol.y[:, -1]
        xA_final = 1.0 - xT_final - xV_final
        
        # Check if Architect dominates at the end
        return xA_final > 0.95  # Recovery threshold from Section 5.2
    
    # Initialize bisection
    V_low, V_high = V_min, V_max
    history = []
    
    # First check endpoints
    outcome_low = run_simulation_and_check(V_low)
    outcome_high = run_simulation_and_check(V_high)
    
    if outcome_low == outcome_high:
        raise ValueError(f"Both endpoints have same outcome ({outcome_low}). "
                        f"Adjust V_min and V_max.")
    
    # Bisection loop
    for iteration in range(max_iter):
        V_mid = (V_low + V_high) / 2.0
        outcome_mid = run_simulation_and_check(V_mid)
        
        history.append((V_mid, outcome_mid, V_low, V_high))
        
        # Update interval
        if outcome_mid == outcome_low:
            V_low = V_mid
            outcome_low = outcome_mid
        else:
            V_high = V_mid
            outcome_high = outcome_mid
        
        # Check convergence
        if V_high - V_low < epsilon:
            break
    
    V_crit = (V_low + V_high) / 2.0
    return V_crit, history


def compute_separatrix_curve(model, xT_values=None, **bisection_kwargs):
    """
    Compute separatrix curve V_crit(xT) for a range of xT values.
    
    Parameters
    ----------
    model : ACEModel
        ACE model instance
    xT_values : array-like, optional
        Array of xT values. If None, uses log-spaced values.
    **bisection_kwargs : dict
        Additional arguments passed to find_separatrix_bisection
        
    Returns
    -------
    results : dict
        Dictionary with xT_values, V_crit_values, and histories
    """
    if xT_values is None:
        # Log-spaced values from 1e-3 to 0.5
        xT_values = np.logspace(-3, np.log10(0.5), 20)
    
    V_crit_values = []
    histories = []
    
    for i, xT in enumerate(xT_values):
        print(f"Computing separatrix for xT = {xT:.4f} ({i+1}/{len(xT_values)})")
        
        try:
            V_crit, history = find_separatrix_bisection(
                model, xT_fixed=xT, **bisection_kwargs
            )
            V_crit_values.append(V_crit)
            histories.append(history)
        except Exception as e:
            print(f"  Error: {e}")
            V_crit_values.append(np.nan)
            histories.append([])
    
    return {
        'xT_values': np.array(xT_values),
        'V_crit_values': np.array(V_crit_values),
        'histories': histories,
        'model_params': {
            'a_tilde': model.a_tilde,
            'b': model.b,
            'gamma': model.gamma,
            'c': model.c,
            'p_star': model.p_star
        }
    }


def save_separatrix_data(results, filename):
    """
    Save separatrix computation results to a CSV file.
    
    Parameters
    ----------
    results : dict
        Results from compute_separatrix_curve
    filename : str
        Output CSV filename
    """
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame({
        'xT': results['xT_values'],
        'V_crit': results['V_crit_values']
    })
    
    # Add metadata as comment lines
    with open(filename, 'w') as f:
        f.write('# Separatrix data for ACE model\n')
        for key, value in results['model_params'].items():
            f.write(f'# {key} = {value}\n')
        
        # Write data
        df.to_csv(f, index=False, lineterminator='\n')
    
    return df