"""
Analysis and sensitivity functions for the ACE model.

Implements the numerical analyses described in Sections 5 and 6 of the manuscript.
"""

import numpy as np
import pandas as pd
from .ode_solver import ACEModel


def compute_stability_landscape(a_tilde_range=(0.5, 2.0), b_range=(0.0, 1.0),
                                gamma_range=(0.0, 1.0), p_star=0.7,
                                n_points=50):
    """
    Compute stability landscape over parameter ranges.
    
    Parameters
    ----------
    a_tilde_range, b_range, gamma_range : tuple
        Parameter ranges (min, max)
    p_star : float
        Fixed Architect symmetry parameter
    n_points : int
        Number of points in each dimension
        
    Returns
    -------
    landscape : dict
        Dictionary containing parameter grids and stability bounds
    """
    # Create parameter grids
    a_tilde_vals = np.linspace(a_tilde_range[0], a_tilde_range[1], n_points)
    b_vals = np.linspace(b_range[0], b_range[1], n_points)
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_points)
    
    # For 2D analysis (b vs gamma) with fixed a_tilde=1.0
    a_tilde_fixed = 1.0
    L_grid = np.zeros((n_points, n_points))
    U_grid = np.zeros((n_points, n_points))
    
    for i, b in enumerate(b_vals):
        for j, gamma in enumerate(gamma_vals):
            model = ACEModel(a_tilde=a_tilde_fixed, b=b, gamma=gamma, 
                           p_star=p_star, c=1.0)  # c is arbitrary here
            L_grid[i, j], U_grid[i, j] = model.L, model.U
    
    return {
        'a_tilde_vals': a_tilde_vals,
        'b_vals': b_vals,
        'gamma_vals': gamma_vals,
        'L_grid': L_grid,
        'U_grid': U_grid,
        'p_star': p_star,
        'a_tilde_fixed': a_tilde_fixed
    }


def compute_policy_feasibility(c_policy=1.0, b_range=(0.0, 1.0),
                               gamma_range=(0.0, 1.0), p_star=0.7,
                               n_points=100):
    """
    Compute policy feasibility map (Figure 2 in manuscript).
    
    Parameters
    ----------
    c_policy : float
        Proposed policy value for c
    b_range, gamma_range : tuple
        Parameter ranges for b and gamma
    p_star : float
        Architect symmetry parameter
    n_points : int
        Number of points in each dimension
        
    Returns
    -------
    feasibility : dict
        Dictionary with feasibility data
    """
    b_vals = np.linspace(b_range[0], b_range[1], n_points)
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_points)
    
    # Create grids
    B, Gamma = np.meshgrid(b_vals, gamma_vals, indexing='ij')
    R = np.zeros_like(B)  # Stability margin
    
    for i in range(n_points):
        for j in range(n_points):
            model = ACEModel(a_tilde=1.0, b=B[i, j], gamma=Gamma[i, j],
                           p_star=p_star, c=c_policy)
            stable, margin = model.check_stability()
            R[i, j] = margin if stable else -1.0
    
    # Create boolean mask for feasible regions
    feasible = R > 0
    
    return {
        'b_grid': B,
        'gamma_grid': Gamma,
        'stability_margin': R,
        'feasible': feasible,
        'c_policy': c_policy,
        'p_star': p_star
    }


def sensitivity_analysis(model, parameter='c', range_pct=0.5, n_points=20):
    """
    Perform sensitivity analysis for a parameter.
    
    Parameters
    ----------
    model : ACEModel
        Baseline model
    parameter : str
        Parameter to vary ('c', 'b', 'gamma', 'a_tilde', 'p_star')
    range_pct : float
        Percentage to vary around baseline (e.g., 0.5 = ±50%)
    n_points : int
        Number of points in sensitivity sweep
        
    Returns
    -------
    sensitivity : dict
        Dictionary with sensitivity results
    """
    # Get baseline value
    baseline_value = getattr(model, parameter)
    
    # Create range
    min_val = baseline_value * (1 - range_pct)
    max_val = baseline_value * (1 + range_pct)
    
    if parameter == 'p_star':
        # p_star must be in (0, 1)
        min_val = max(0.01, min_val)
        max_val = min(0.99, max_val)
    
    param_values = np.linspace(min_val, max_val, n_points)
    
    # Store results
    results = {
        'parameter': parameter,
        'values': param_values,
        'L_values': [],
        'U_values': [],
        'stable': [],
        'width': []
    }
    
    for val in param_values:
        # Create new model with varied parameter
        model_kwargs = {
            'a_tilde': model.a_tilde,
            'b': model.b,
            'gamma': model.gamma,
            'c': model.c,
            'p_star': model.p_star
        }
        model_kwargs[parameter] = val
        
        new_model = ACEModel(**model_kwargs)
        L, U = new_model.L, new_model.U
        stable = L < new_model.c < U
        width = U - L
        
        results['L_values'].append(L)
        results['U_values'].append(U)
        results['stable'].append(stable)
        results['width'].append(width)
    
    # Convert to arrays
    for key in ['L_values', 'U_values', 'stable', 'width']:
        results[key] = np.array(results[key])
    
    return results


def verify_solver_convergence(model, methods=('RK45', 'BDF'), 
                              rtol_values=(1e-6, 1e-8, 1e-10),
                              y0=None):
    """
    Verify ODE solver convergence (Section 5.3 checks).
    
    Parameters
    ----------
    model : ACEModel
        ACE model instance
    methods : tuple
        Solver methods to compare
    rtol_values : tuple
        Relative tolerance values to test
    y0 : array-like, optional
        Initial conditions
        
    Returns
    -------
    convergence : dict
        Dictionary with convergence test results
    """
    if y0 is None:
        y0 = [0.01, 0.01, 0.5]
    
    results = {}
    
    for method in methods:
        results[method] = {}
        
        for rtol in rtol_values:
            sol = model.simulate(method=method, rtol=rtol, y0=y0)
            
            # Store final state
            final_state = sol.y[:, -1]
            results[method][rtol] = {
                'success': sol.success,
                'final_state': final_state,
                'n_eval': sol.nfev,
                'message': sol.message
            }
    
    # Compare results
    comparisons = []
    for method1 in methods:
        for method2 in methods:
            if method1 != method2:
                diff = np.max(np.abs(
                    results[method1][1e-8]['final_state'] - 
                    results[method2][1e-8]['final_state']
                ))
                comparisons.append({
                    'methods': f'{method1} vs {method2}',
                    'max_difference': diff
                })
    
    return {
        'solver_results': results,
        'comparisons': comparisons
    }