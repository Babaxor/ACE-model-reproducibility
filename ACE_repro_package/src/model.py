"""
Core ACE model definition: payoffs, dynamics, and parameter management.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class ACEModel:
    """
    Architect in Coupled Ecosystem (ACE) Model.
    
    A three-strategy evolutionary game coupled unidirectionally to a
    logistic resource stock.
    
    Parameters
    ----------
    a_tilde : float
        Effective extraction rate (must be > 0)
    b : float
        Cooperation cost coefficient (≥ 0)
    gamma : float
        Responsibility benefit coefficient (≥ 0)
    c : float
        Self-cost intensity (must be in (L, U) for stability)
    p_star : float
        Architect symmetry parameter (0 < p_star < 1)
    r : float
        Resource intrinsic growth rate (> 0)
    lam : float
        Resource maintenance threshold (> 0)
    K : float
        Resource carrying capacity (> 0)
    """
    
    def __init__(self, a_tilde: float = 1.0, b: float = 0.2, 
                 gamma: float = 0.5, c: float = 1.0, p_star: float = 0.7,
                 r: float = 1.0, lam: float = 0.3, K: float = 1.0):
        
        # Store parameters
        self.params = {
            'a_tilde': a_tilde,
            'b': b,
            'gamma': gamma,
            'c': c,
            'p_star': p_star,
            'r': r,
            'lam': lam,
            'K': K
        }
        
        # Validate parameters
        self._validate_parameters()
        
        # Compute derived quantities
        self.L, self.U = self.compute_stability_bounds()
        self.V_star = self.compute_resource_equilibrium()
    
    def _validate_parameters(self) -> None:
        """Validate model parameters."""
        if self.params['a_tilde'] <= 0:
            raise ValueError("a_tilde must be > 0")
        if not (0 < self.params['p_star'] < 1):
            raise ValueError("p_star must be in (0, 1)")
        if self.params['b'] < 0 or self.params['gamma'] < 0:
            raise ValueError("b and gamma must be ≥ 0")
        if self.params['r'] <= 0 or self.params['K'] <= 0:
            raise ValueError("r and K must be > 0")
    
    def compute_stability_bounds(self) -> Tuple[float, float]:
        """
        Compute stability bounds L and U (Theorem 3.3).
        
        Returns
        -------
        L, U : tuple
            Lower and upper bounds for stability
        """
        p = self.params['p_star']
        a_tilde = self.params['a_tilde']
        b = self.params['b']
        gamma = self.params['gamma']
        
        # Lower bound L
        numerator_L = (a_tilde * (1 - p)**2 + 
                      b * p**2 + 
                      gamma * p * (1 - p))
        denominator_L = 1 - p + p**2
        L = numerator_L / denominator_L
        
        # Upper bound U
        U = (a_tilde + 
             b / (1 - p) + 
             gamma * p / (1 - p))
        
        return L, U
    
    def compute_resource_equilibrium(self) -> float:
        """
        Compute resource equilibrium V*.
        
        Returns
        -------
        V_star : float
            Resource equilibrium value
        """
        p = self.params['p_star']
        r = self.params['r']
        lam = self.params['lam']
        K = self.params['K']
        
        if p > lam / r:
            return K * (p - lam / r)
        else:
            return 0.0
    
    def payoff_matrix(self) -> Dict[str, float]:
        """
        Compute 3x3 payoff matrix (Appendix D).
        
        Returns
        -------
        payoffs : dict
            Dictionary of payoff entries
        """
        p = self.params['p_star']
        a_tilde = self.params['a_tilde']
        b = self.params['b']
        gamma = self.params['gamma']
        c = self.params['c']
        
        # Against Tyrant (T)
        Pi_TT = -c
        Pi_VT = 0
        Pi_AT = -c * p * (1 - p)
        
        # Against Victim (V)
        Pi_TV = a_tilde + gamma - c
        Pi_VV = -b
        Pi_AV = a_tilde * p - b * p + gamma * p - c * p * (1 - p)
        
        # Against Architect (A)
        Pi_TA = a_tilde * (1 - p) + gamma * p - c
        Pi_VA = -b * p
        Pi_AA = (a_tilde * p * (1 - p) - 
                b * p**2 + 
                gamma * p**2 - 
                c * p * (1 - p))
        
        return {
            'TT': Pi_TT, 'TV': Pi_TV, 'TA': Pi_TA,
            'VT': Pi_VT, 'VV': Pi_VV, 'VA': Pi_VA,
            'AT': Pi_AT, 'AV': Pi_AV, 'AA': Pi_AA
        }
    
    def fitness_functions(self, xT: float, xV: float) -> Dict[str, float]:
        """
        Compute fitness functions for each strategy.
        
        Parameters
        ----------
        xT, xV : float
            Frequencies of Tyrant and Victim strategies
        
        Returns
        -------
        fitness : dict
            Fitness values for T, V, A
        """
        xA = 1.0 - xT - xV
        payoffs = self.payoff_matrix()
        
        f_T = (xT * payoffs['TT'] + 
               xV * payoffs['TV'] + 
               xA * payoffs['TA'])
        
        f_V = (xT * payoffs['VT'] + 
               xV * payoffs['VV'] + 
               xA * payoffs['VA'])
        
        f_A = (xT * payoffs['AT'] + 
               xV * payoffs['AV'] + 
               xA * payoffs['AA'])
        
        return {'T': f_T, 'V': f_V, 'A': f_A}
    
    def average_responsibility(self, xT: float, xV: float) -> float:
        """
        Compute average responsibility R̄.
        
        Parameters
        ----------
        xT, xV : float
            Strategy frequencies
        
        Returns
        -------
        R_bar : float
            Average responsibility
        """
        xA = 1.0 - xT - xV
        return xV + self.params['p_star'] * xA
    
    def check_stability(self, c: Optional[float] = None) -> Tuple[bool, float]:
        """
        Check if given c is in stable interval.
        
        Parameters
        ----------
        c : float, optional
            Self-cost parameter. If None, uses model's c.
            
        Returns
        -------
        stable : bool
            True if c is between L and U
        margin : float
            Minimum distance to bounds
        """
        if c is None:
            c = self.params['c']
        
        stable = self.L < c < self.U
        if stable:
            margin = min(c - self.L, self.U - c)
        else:
            margin = 0.0
        
        return stable, margin
    
    def __repr__(self) -> str:
        """String representation of the model."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"ACEModel({params_str})"