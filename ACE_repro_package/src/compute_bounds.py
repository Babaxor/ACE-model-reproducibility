"""
Functions for computing stability bounds L and U.
"""

import numpy as np
from typing import Tuple, Dict


def compute_LU(a_tilde: float, b: float, gamma: float, 
               p_star: float) -> Tuple[float, float, Dict]:
    """
    Compute stability bounds L and U (Theorem 3.3).
    
    Parameters
    ----------
    a_tilde : float
        Effective extraction rate
    b : float
        Cooperation cost coefficient
    gamma : float
        Responsibility benefit coefficient
    p_star : float
        Architect symmetry parameter
        
    Returns
    -------
    L, U : tuple
        Lower and upper bounds
    info : dict
        Additional information about the computation
    """
    # Validate inputs
    if a_tilde <= 0:
        raise ValueError("a_tilde must be > 0")
    if not (0 < p_star < 1):
        raise ValueError("p_star must be in (0, 1)")
    if b < 0 or gamma < 0:
        raise ValueError("b and gamma must be ≥ 0")
    
    # Compute L
    numerator_L = (a_tilde * (1 - p_star)**2 + 
                  b * p_star**2 + 
                  gamma * p_star * (1 - p_star))
    denominator_L = 1 - p_star + p_star**2
    L = numerator_L / denominator_L
    
    # Compute U
    U = (a_tilde + 
         b / (1 - p_star) + 
         gamma * p_star / (1 - p_star))
    
    # Compute width
    width = U - L
    
    # Boundary limits
    limits = {
        'p→0': {'L': a_tilde, 'U': a_tilde + b},
        'p→1': {'L': b, 'U': float('inf')}
    }
    
    info = {
        'parameters': {
            'a_tilde': a_tilde,
            'b': b,
            'gamma': gamma,
            'p_star': p_star
        },
        'intermediate': {
            'numerator_L': numerator_L,
            'denominator_L': denominator_L,
            'width': width
        },
        'limits': limits,
        'nonempty': width > 0
    }
    
    return L, U, info


def compute_stability_region_grid(a_tilde_range: Tuple[float, float] = (0.5, 2.0),
                                 b_range: Tuple[float, float] = (0.0, 1.0),
                                 gamma_range: Tuple[float, float] = (0.0, 1.0),
                                 p_star: float = 0.7,
                                 n_points: int = 50) -> Dict:
    """
    Compute stability region over parameter grids.
    
    Parameters
    ----------
    a_tilde_range, b_range, gamma_range : tuple
        Parameter ranges
    p_star : float
        Fixed Architect symmetry
    n_points : int
        Grid resolution
        
    Returns
    -------
    results : dict
        Grids and stability information
    """
    # Create grids
    a_tilde_vals = np.linspace(a_tilde_range[0], a_tilde_range[1], n_points)
    b_vals = np.linspace(b_range[0], b_range[1], n_points)
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_points)
    
    # Initialize result arrays
    L_grid = np.zeros((n_points, n_points, n_points))
    U_grid = np.zeros((n_points, n_points, n_points))
    width_grid = np.zeros((n_points, n_points, n_points))
    
    # Compute over grid
    for i, a in enumerate(a_tilde_vals):
        for j, b_val in enumerate(b_vals):
            for k, g in enumerate(gamma_vals):
                L, U, _ = compute_LU(a, b_val, g, p_star)
                L_grid[i, j, k] = L
                U_grid[i, j, k] = U
                width_grid[i, j, k] = U - L
    
    return {
        'a_tilde_grid': a_tilde_vals,
        'b_grid': b_vals,
        'gamma_grid': gamma_vals,
        'L_grid': L_grid,
        'U_grid': U_grid,
        'width_grid': width_grid,
        'p_star': p_star
    }


def check_stability_for_c(c: float, a_tilde: float, b: float, 
                         gamma: float, p_star: float) -> Dict:
    """
    Check stability for a specific c value.
    
    Parameters
    ----------
    c : float
        Self-cost parameter to check
    a_tilde, b, gamma, p_star : float
        Model parameters
        
    Returns
    -------
    result : dict
        Stability information
    """
    L, U, info = compute_LU(a_tilde, b, gamma, p_star)
    
    stable = L < c < U
    margin = min(c - L, U - c) if stable else 0.0
    
    return {
        'c': c,
        'L': L,
        'U': U,
        'stable': stable,
        'stability_margin': margin,
        'within_bounds': {
            'above_L': c > L,
            'below_U': c < U
        },
        'distance_to_bounds': {
            'to_L': c - L,
            'to_U': U - c
        }
    }